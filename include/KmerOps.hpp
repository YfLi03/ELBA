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

struct BatchSender
{
    std::shared_ptr<CommGrid> commgrid;
    size_t bytes_per_proc;
    size_t send_threshold;
    uint8_t* sendbuf;
    uint8_t* recvbuf;

    BatchSender(std::shared_ptr<CommGrid> commgrid, size_t bytes_per_proc, size_t send_threshold, uint8_t* sendbuf, uint8_t* recvbuf) : commgrid(commgrid), bytes_per_proc(bytes_per_proc), send_threshold(send_threshold), sendbuf(sendbuf), recvbuf(recvbuf) {}

    void send_all()
    {
        int nprocs = commgrid->GetSize();
        int myrank = commgrid->GetRank();

        if ( bytes_per_proc <= send_threshold ) {
            MPI_ALLTOALL(sendbuf, bytes_per_proc, MPI_BYTE, recvbuf, bytes_per_proc, MPI_BYTE, commgrid->GetWorld());
            return;
        }
        int times = (bytes_per_proc - 1) / send_threshold + 1;

        uint8_t* sendtmp = new uint8_t[send_threshold * nprocs];
        uint8_t* recvtmp = new uint8_t[send_threshold * nprocs];

        for(int i = 0; i < times; i++){
            MPI_Count_type send_size = std::min(send_threshold, bytes_per_proc - i * send_threshold);
            
            for(int j = 0; j < nprocs; j++){
                memcpy(sendtmp + j * send_size, sendbuf + i * send_threshold + j * bytes_per_proc, send_size);
            }

            MPI_ALLTOALL(sendtmp, send_size, MPI_BYTE, recvtmp, send_size, MPI_BYTE, commgrid->GetWorld());

            for(int j = 0; j < nprocs; j++){
                memcpy(recvbuf + i * send_threshold + j * bytes_per_proc, recvtmp + j * send_size, send_size);
            }
        }

        delete[] sendtmp;
        delete[] recvtmp;
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
*/

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

#endif // ELBA_KMEROPS_HPP
