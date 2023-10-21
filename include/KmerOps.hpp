#ifndef ELBA_KMEROPS_HPP
#define ELBA_KMEROPS_HPP

#include "common.h"
#include "Kmer.hpp"
#include "DnaSeq.hpp"
#include "DnaBuffer.hpp"

#ifndef MAX_ALLTOALL_MEM
#define MAX_ALLTOALL_MEM (128ULL * 1024ULL * 1024ULL * 1024ULL)
#endif


typedef uint32_t PosInRead;
typedef  int64_t ReadId;

typedef std::array<PosInRead, UPPER_KMER_FREQ> POSITIONS;
typedef std::array<ReadId,    UPPER_KMER_FREQ> READIDS;
// typedef std::tuple<TKmer, ReadId, PosInRead> KmerSeed;

typedef std::tuple<TKmer, READIDS, POSITIONS, int> KmerListEntry;
typedef std::vector<KmerListEntry> KmerList;

#define DEFAULT_THREAD_PER_TASK 4
#define MAX_THREAD_MEMORY_BOUNDED 16

/* yfli: This struct stores the same information as KmerSeed */
/* I need this struct because the GetByte() member function is necessary for sorting */
struct KmerSeedStruct{
    TKmer kmer;      
    ReadId readid;
    PosInRead posinread;

    KmerSeedStruct(TKmer kmer, ReadId readid, PosInRead posinread) : kmer(kmer), readid(readid), posinread(posinread) {};
    KmerSeedStruct(const KmerSeedStruct& o) : kmer(o.kmer), readid(o.readid), posinread(o.posinread) {};
    // KmerSeedStruct(const KmerSeed& o) : kmer(std::get<0>(o)), readid(std::get<1>(o)), posinread(std::get<2>(o)) {};
    KmerSeedStruct(KmerSeedStruct&& o) :    // yfli: Not sure if it's appropriate to use std::move() here
        kmer(std::move(o.kmer)), readid(std::move(o.readid)), posinread(std::move(o.posinread)) {};
    KmerSeedStruct() {};

    int GetByte(int &i) const
    {
        return kmer.getByte(i);
    }

    bool operator<(const KmerSeedStruct& o) const
    {
        return kmer < o.kmer;
    }

    bool operator==(const KmerSeedStruct& o) const
    {
        return kmer == o.kmer;
    }

    bool operator!=(const KmerSeedStruct& o) const
    {
        return kmer != o.kmer;
    }

    KmerSeedStruct& operator=(const KmerSeedStruct& o)
    {
        kmer = o.kmer;
        readid = o.readid;
        posinread = o.posinread;
        return *this;
    }
};

typedef std::vector<std::vector<KmerSeedStruct>> KmerSeedBuckets;

std::unique_ptr<CT<PosInRead>::PSpParMat>
create_kmer_matrix(const DnaBuffer& myreads, const KmerList& kmerlist, std::shared_ptr<CommGrid> commgrid);

std::unique_ptr<KmerSeedBuckets> 
exchange_kmer(const DnaBuffer& myreads,
     std::shared_ptr<CommGrid> commgrid,
     int thr_per_task = DEFAULT_THREAD_PER_TASK,
     int max_thr_membounded = MAX_THREAD_MEMORY_BOUNDED);

std::unique_ptr<KmerList>
filter_kmer(std::unique_ptr<KmerSeedBuckets>& recv_kmerseeds, 
     std::shared_ptr<CommGrid> commgrid, 
     int thr_per_task = DEFAULT_THREAD_PER_TASK);


int GetKmerOwner(const TKmer& kmer, int nprocs);

/*
struct BatchState
{
    std::shared_ptr<CommGrid> commgrid;
    size_t mynumreads;
    size_t memthreshold;
    size_t mykmerssofar;
    size_t mymaxsending;
    ReadId myreadid;

    BatchState(size_t mynumreads, std::shared_ptr<CommGrid> commgrid) : commgrid(commgrid), mynumreads(mynumreads), memthreshold((MAX_ALLTOALL_MEM / commgrid->GetSize()) << 1), mykmerssofar(0), mymaxsending(0), myreadid(0) {}

    bool ReachedThreshold(const size_t len)
    {
        return (mymaxsending * TKmer::NBYTES >= memthreshold || (mykmerssofar + len) * TKmer::NBYTES >= MAX_ALLTOALL_MEM);
    }

    bool Finished() const
    {
        int alldone;
        int imdone = !!(myreadid >= static_cast<ReadId>(mynumreads));
        MPI_Allreduce(&imdone, &alldone, 1, MPI_INT, MPI_LAND, commgrid->GetWorld());
        return bool(alldone);
    }
};
*/

class BatchSender
{
public:
    enum Status
    {
        BATCH_NOT_INIT = 0,
        BATCH_SENDING = 1,
        BATCH_DONE = 2
    };

private:
    std::shared_ptr<CommGrid> commgrid;
    int nprocs;
    int myrank;

    Status status;
    MPI_Request req;

    size_t bytes_per_proc;
    size_t batch_sz;        /* batch size in bytes */
    size_t next_loc;        /* next location for sending */
    size_t current_sz;      /* current valid size of the recvbuf */

    uint8_t* sendbuf;

    /* x is always the one that is currently sending, y is the one that can be used */
    uint8_t* sendtmp_x;
    uint8_t* recvtmp_x;
    uint8_t* sendtmp_y;
    uint8_t* recvtmp_y;


public:

    BatchSender(std::shared_ptr<CommGrid> commgrid, size_t bytes_per_proc, size_t batch_size, uint8_t* sendbuf, size_t seedbytes) : 
        commgrid(commgrid), bytes_per_proc(bytes_per_proc), 
        batch_sz(batch_size), sendbuf(sendbuf), status(BATCH_NOT_INIT),
        next_loc(0), current_sz(0)
    {
        batch_sz = std::min(batch_sz, bytes_per_proc);
        if (batch_sz % seedbytes != 0) {
            batch_sz = (batch_sz / seedbytes) * seedbytes;
        }

        nprocs = commgrid->GetSize();
        myrank = commgrid->GetRank();

        /* one send tmp buf should be enough. I'm using two anyway. */
        sendtmp_x = new uint8_t[batch_sz * nprocs];
        recvtmp_x = new uint8_t[batch_sz * nprocs];

        sendtmp_y = nullptr;
        recvtmp_y = nullptr;
    };

    ~BatchSender()  
    {
        if (sendtmp_x != nullptr) delete[] sendtmp_x;
        if (recvtmp_y != nullptr) delete[] recvtmp_x;
        if (sendtmp_y != nullptr) delete[] sendtmp_y;
        if (recvtmp_y != nullptr) delete[] recvtmp_y;
        
    }

    // constructor (the vars, strategy)
    // destructor (delete the tmp buffers)
    // init (init the tmp buffers, start first sending)
    // progress (wait for last send to complete and start next send if necessary)

    Status get_status() { return status; }

    uint8_t* get_recvbuf() { return recvtmp_y; }

    void init()
    {
        status = BATCH_SENDING;

        if (bytes_per_proc <= batch_sz) {
            MPI_Ialltoall(sendbuf, bytes_per_proc, MPI_BYTE, recvtmp_x, bytes_per_proc, MPI_BYTE, commgrid->GetWorld(), &req);
            next_loc = bytes_per_proc;
            current_sz = bytes_per_proc;
            return;
        }

        sendtmp_y = new uint8_t[batch_sz * nprocs];
        recvtmp_y = new uint8_t[batch_sz * nprocs];

        for (int i = 0; i < nprocs; i++) {
            memcpy(sendtmp_x + batch_sz * i, sendbuf + bytes_per_proc * i, batch_sz);
        }

        MPI_Ialltoall(sendtmp_x, batch_sz, MPI_BYTE, recvtmp_x, batch_sz, MPI_BYTE, commgrid->GetWorld(), &req);
        next_loc = batch_sz;
        current_sz = batch_sz;
    }

    size_t progress() {
        if (status != BATCH_SENDING) return 0;

        MPI_Wait(&req, MPI_STATUS_IGNORE);

        std::swap(sendtmp_x, sendtmp_y);
        std::swap(recvtmp_x, recvtmp_y);

        if (myrank==0){
            std::cout<<(size_t)sendtmp_x<<" "<<(size_t)sendtmp_y<<std::endl;
            std::cout<<(size_t)recvtmp_x<<" "<<(size_t)recvtmp_y<<std::endl;
        }

        // send completed
        if (next_loc >= bytes_per_proc) {
            status = BATCH_DONE;
            return current_sz;
        }

        // send not completed

        // this can happen at the beginning to overlap more, but it'll be more complex
        size_t send_size = std::min(batch_sz, bytes_per_proc - next_loc);
        for (int i = 0; i < nprocs; i++) {
            memcpy(sendtmp_x + i * send_size, sendbuf + next_loc + i * bytes_per_proc, send_size);
        }

        MPI_Ialltoall(sendtmp_x, send_size, MPI_BYTE, recvtmp_x, send_size, MPI_BYTE, commgrid->GetWorld(), &req);
        next_loc += send_size;
        size_t lsize = current_sz;
        current_sz = send_size;

        return lsize;
    }
};

struct BatchStorer
{
    KmerSeedBuckets& kmerseeds;
    int taskcnt, nprocs;
    size_t seedbytes;
    size_t* expected_recvcnt;
    size_t* recvcnt;
    int* recv_task_id;

    BatchStorer(KmerSeedBuckets& kmerseeds, size_t seedbytes, int taskcnt, int nprocs, size_t* expected_recvcnt) : 
        kmerseeds(kmerseeds), seedbytes(seedbytes), taskcnt(taskcnt), nprocs(nprocs), expected_recvcnt(expected_recvcnt)
        {
            recvcnt = new size_t[nprocs];
            recv_task_id = new int[nprocs];
            memset(recvcnt, 0, sizeof(size_t) * nprocs);
            memset(recv_task_id, 0, sizeof(int) * nprocs);
        }

    ~BatchStorer()
    {
        
        delete[] recvcnt;
        delete[] recv_task_id;
        
    }

    void store(uint8_t* recvbuf, size_t recv_size)
    {
        for(int i = 0; i < nprocs; i++) 
        {
            if (recv_task_id[i] >= taskcnt) continue;
            int working_task = recv_task_id[i];
            size_t cnt = recvcnt[i];
            size_t max_cnt = expected_recvcnt[ taskcnt * i + working_task ];

            uint8_t* addrs2read = recvbuf + recv_size * i;

            for (size_t j = 0; j < recv_size / seedbytes; j++) {

                TKmer kmer((void*)addrs2read);                
                ReadId readid = *((ReadId*)(addrs2read + TKmer::NBYTES));
                PosInRead pos = *((PosInRead*)(addrs2read + TKmer::NBYTES + sizeof(ReadId)));
                kmerseeds[working_task].emplace_back(kmer, readid, pos);

                addrs2read += seedbytes;
                cnt++;
                if (cnt >= max_cnt) {
                    recv_task_id[i]++;
                    if (recv_task_id[i] >= taskcnt) break;
                    working_task++;
                    cnt = 0;
                    max_cnt = expected_recvcnt[ taskcnt * i + working_task ];
                }

            }
            recvcnt[i] = cnt;

        }
    }
};

// may need that in the future
/*
struct KmerEstimateHandler
{
    HyperLogLog& hll;

    KmerEstimateHandler(HyperLogLog& hll) : hll(hll) {}

    void operator()(const TKmer& kmer, size_t kid, size_t rid)
    {
        auto s = kmer.GetString();
        hll.add(s.c_str());
    }
};


struct KmerPartitionHandler
{
    int nthreads_tot;
    std::vector<std::vector<TKmer>>& kmerbuckets;

    KmerPartitionHandler(std::vector<std::vector<TKmer>>& kmerbuckets) : nthreads_tot(kmerbuckets.size()), kmerbuckets(kmerbuckets) {}

    void operator()(const TKmer& kmer, size_t kid, size_t rid)
    {
        kmerbuckets[GetKmerOwner(kmer, nthreads_tot)].push_back(kmer);
    }

    void operator()(const TKmer& kmer, BatchState& state, size_t kid)
    {
        auto& kmerbucket = kmerbuckets[GetKmerOwner(kmer, nthreads_tot)];
        kmerbucket.push_back(kmer);
        state.mymaxsending = std::max(kmerbucket.size(), state.mymaxsending);
    }
};
*/


struct KmerParserHandler
{
    int nprocs;
    ReadId readoffset;
    std::vector<std::vector<KmerSeedStruct>>& kmerseeds;

    KmerParserHandler(std::vector<std::vector<KmerSeedStruct>>& kmerseeds, ReadId readoffset) : nprocs(kmerseeds.size()), readoffset(readoffset), kmerseeds(kmerseeds) {}

    void operator()(const TKmer& kmer, size_t kid, size_t rid)
    {
        kmerseeds[GetKmerOwner(kmer, nprocs)].emplace_back(kmer, static_cast<ReadId>(rid) + readoffset, static_cast<PosInRead>(kid));
    }
};

template <typename KmerHandler>
void ForeachKmer(const DnaBuffer& myreads, KmerHandler& handler)
{
    size_t i;

    /*
     * Go through each local read.
     */
    for (i = 0; i < myreads.size(); ++i)
    {
        /*
         * If it is too small then continue to the next one.
         */
        if (myreads[i].size() < KMER_SIZE)
            continue;

        /*
         * Get all the representative k-mer seeds.
         */
        std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);

        size_t j = 0;

        /*
         * Go through each k-mer seed.
         */
        for (auto meritr = repmers.begin(); meritr != repmers.end(); ++meritr, ++j)
        {
            handler(*meritr, j, i);
        }
    }
}

template <typename KmerHandler>
void ForeachKmerParallel(const DnaBuffer& myreads, std::vector<KmerHandler>& handlers, int nthreads)
{
    assert(nthreads > 0);

    /*
     * Go through each local read.
     */

    // yfli: TODO: test the performance of different number of threads. This might be a memory bandwidth issue.

    #pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < myreads.size(); ++i)
    {
        int tid = omp_get_thread_num();
        /*
         * If it is too small then continue to the next one.
         */
        if (myreads[i].size() < KMER_SIZE)
            continue;

        /*
         * Get all the representative k-mer seeds.
         */
        std::vector<TKmer> repmers = TKmer::GetRepKmers(myreads[i]);

        size_t j = 0;

        /*
         * Go through each k-mer seed.
         */
        for (auto meritr = repmers.begin(); meritr != repmers.end(); ++meritr, ++j)
        {
            handlers[tid](*meritr, j, i);
        }
    }
}

/*
template <typename KmerHandler>
void ForeachKmer(const DnaBuffer& myreads, KmerHandler& handler, BatchState& state)
{
    for (; state.myreadid < static_cast<ReadId>(myreads.size()); state.myreadid++)
    {
        const DnaSeq& sequence = myreads[state.myreadid];

        if (sequence.size() < KMER_SIZE)
            continue;

        std::vector<TKmer> repmers = TKmer::GetRepKmers(sequence);
        state.mykmerssofar += repmers.size();

        size_t j = 0;

        for (auto meritr = repmers.begin(); meritr != repmers.end(); ++meritr, ++j)
        {
            handler(*meritr, state, j);
        }

        if (state.ReachedThreshold(sequence.size()))
        {
            state.myreadid++;
            return;
        }
    }
}
*/
#endif // ELBA_KMEROPS_HPP
