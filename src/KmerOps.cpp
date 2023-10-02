
#include "KmerOps.hpp"
#include "Logger.hpp"
#include "DnaSeq.hpp"
#include "MPITimer.hpp"
#include "ParadisSort.hpp"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include <omp.h>


template<int bytes, typename T>
void radix_sort(T* st, T* ed) {

  int S_GROUP = (bytes * 8 + S_BIT - 1) / S_BIT;
  // assert(S_GROUP == 4);
  // assert(S_DEC == 65536);

  // std::cout<<"aaa"<<std::endl;
  T* buf = new T[ed-st];
  size_t* cnt = new size_t[S_DEC];
  // std::cout<<cnt<<std::endl;
  // std::cout<<"bbb"<<std::endl;

  for(int i=0; i<S_GROUP; i++){

    for(int j=0; j<S_DEC; j++)
      cnt[j] = 0;
  // std::cout<<" ccc"<<std::endl;

    for(T* j=st; j<ed; j++){
     // assert(j->GetBytes(i) < S_DEC);
      ++cnt[j->GetBytes(i)];
      // std::cout<<j->GetBytes(i)<<std::endl;
     // assert(cnt[j->GetBytes(i)] < (ed-st));

    }
  // std::cout<<"ddd"<<std::endl;

    for(int j=1; j<S_DEC; j++)
      cnt[j] += cnt[j-1];
  // std::cout<<"eee"<<std::endl;

    for(T* j=ed-1; j>=st; j--){
       // assert(j->GetBytes(i) < S_DEC);
       // assert(cnt[j->GetBytes(i)] > 0);
       //  assert(cnt[j->GetBytes(i)] <= (ed-st));
        buf[--cnt[j->GetBytes(i)]] = *j;
    }
      
  // std::cout<<"xxx"<<std::endl;

    memcpy(st, buf, (ed-st)*sizeof(T));
  }
  // std::cout<<"zzz"<<std::endl;

  delete[] buf;
  delete[] cnt;
}

std::unique_ptr<KmerList>
get_kmer_list(const DnaBuffer& myreads, std::shared_ptr<CommGrid> commgrid)
{
    MPI_Comm comm = MPI_COMM_WORLD;

    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();
    bool single_node = (nprocs == 1);

    Logger logger(commgrid);
    std::ostringstream rootlog;
#define per_thread 16

    int nthreads = omp_get_max_threads() / per_thread;                         // yfli: is not working well
    logger() << nthreads << std::endl;
    logger.Flush("max_thread_num");


    size_t numreads = myreads.size();                              /* Number of locally stored reads */

    MPITimer timer(comm);
    timer.start();

    size_t readoffset = numreads;                                   // yfli: not sure if we need this initial value
    if (!single_node)
        MPI_Exscan(&numreads, &readoffset, 1, MPI_SIZE_T, MPI_SUM, commgrid->GetWorld());
    if (!myrank) readoffset = 0;    

    std::vector<std::vector<KmerSeed>> kmerseeds(nprocs * nthreads);
    logger.Flush("A");


    /* prepare the vars for each thread. each **thread** is responsible for some reeds */
    std::vector<std::vector<std::vector<KmerSeed>>> kmerseeds_vecs;
    std::vector<KmerParserHandler> parser_vecs;
    for (int i = 0; i < std::min(nthreads, MAX_THREAD_MEMORY_BOUNDED); i++) {
        kmerseeds_vecs.push_back(std::vector<std::vector<KmerSeed>>(nprocs * nthreads));
    }
    logger.Flush("B");

    for (int i = 0; i < std::min(nthreads, MAX_THREAD_MEMORY_BOUNDED); i++ ) {
        /* This is a little bit tricky, as we need to avoid the kmerseeds_vecs from reallocating */
        parser_vecs.push_back(KmerParserHandler(kmerseeds_vecs[i], static_cast<ReadId>(readoffset)));
    }

    logger.Flush("C");


    ForeachKmerParallel(myreads, parser_vecs, nthreads);

    logger.Flush("D");


    // yfli: maybe try parallelize this
    /* merge the seed vecs into one */
    for (int j = 0; j < nprocs * nthreads; j++) {
        for (int i = 0; i < std::min(nthreads, MAX_THREAD_MEMORY_BOUNDED); i++) {
            kmerseeds[j].insert(kmerseeds[j].end(), kmerseeds_vecs[i][j].begin(), kmerseeds_vecs[i][j].end());
        }
    }
    timer.stop_and_log("Local K-mer parsing");

    /* yfli: 
     * generally speaking, it's suggested to use 1 process per node for 1 node
     * and 4 process per node for more than 1 node
     */
    std::vector<std::vector<KmerSeedStruct>> recv_kmerseeds(nthreads);
    std::vector<MPI_Count_type> numkmerseeds_thread(nthreads);


    if (nprocs == 1){
        // we're definitely wasting some time here
        // consider switching to the KMerSeedStruct completely
        timer.start();
        for (int i = 0; i < nthreads; i++) {
            // yfli: will this implicit type conversion work?
            recv_kmerseeds[i].insert(recv_kmerseeds[i].end(), kmerseeds[i].begin(), kmerseeds[i].end());
            numkmerseeds_thread[i] = kmerseeds[i].size();
        }
        timer.stop_and_log("Local K-mer format conversion (nprocs == 1) ");
    } 
    else
    {
        timer.start();

        std::vector<MPI_Count_type> sendcnt(nprocs); //, recvcnt(nprocs);
        std::vector<MPI_Displ_type> sdispls(nprocs), rdispls(nprocs);
        std::vector<MPI_Count_type> sthrcnt(nprocs * nthreads), rthrcnt(nprocs * nthreads);  // cnt for each thread

        constexpr size_t seedbytes = TKmer::NBYTES + sizeof(ReadId) + sizeof(PosInRead);

        #if LOG_LEVEL >= 2
        logger() << std::setprecision(4) << "sending 'row' k-mers to each processor in this amount (megabytes): {";
        #endif

        for (int i = 0; i < nprocs; ++i)
        {
            for (int j = 0; j < nthreads; j++) {
                sendcnt[i] += kmerseeds[i * nthreads + j].size() * seedbytes;
                sthrcnt[i * nthreads + j] = kmerseeds[i * nthreads + j].size() * seedbytes;
            }
            
            #if LOG_LEVEL >= 2
            logger() << (static_cast<double>(sendcnt[i]) / (1024 * 1024)) << ",";
            #endif
        }

        #if LOG_LEVEL >= 2
        logger() << "}";
        logger.Flush("K-mer exchange sendcounts:");
        #endif

        MPI_Count_type max_send = *std::max_element(sendcnt.begin(), sendcnt.end());
        MPI_Count_type buf_size = 0;

        MPI_Allreduce(&max_send, &buf_size, 1, MPI_COUNT_TYPE, MPI_MAX, commgrid->GetWorld());

        MPI_Count_type total_buf = buf_size * nprocs;

        // exchange the send/recv cnt information to prepare for data transfer
        // MPI_ALLTOALL(sendcnt.data(), 1, MPI_COUNT_TYPE, recvcnt.data(), 1, MPI_COUNT_TYPE, commgrid->GetWorld());
        MPI_ALLTOALL(sthrcnt.data(), nthreads, MPI_COUNT_TYPE, rthrcnt.data(), nthreads, MPI_COUNT_TYPE, commgrid->GetWorld());

        // copying data to the sendbuf
        // yfli: maybe these memcpy can be combined into fewer operations by adjusting the data structure
        // yfli: we're doing many times of redandant memcpy here actually, maybe try avoid this if performance is not good
        std::vector<uint8_t> sendbuf(total_buf, 0);

        omp_set_num_threads(std::min(nthreads, MAX_THREAD_MEMORY_BOUNDED));
        # pragma omp parallel for schedule(static)
        for (int i = 0; i < nprocs; ++i)
        {
            assert(kmerseeds[i].size() == (sendcnt[i] / seedbytes));
            uint8_t *addrs2fill = sendbuf.data() + i * buf_size;

            for (int j = 0; j < nthreads; j++) {
                for (MPI_Count_type k = 0; k < kmerseeds[i * nthreads + j].size(); ++k)
                {
                    auto& seeditr = kmerseeds[i * nthreads + j][k];
                    TKmer kmer = std::get<0>(seeditr);
                    ReadId readid = std::get<1>(seeditr);
                    PosInRead pos = std::get<2>(seeditr);
                    memcpy(addrs2fill, kmer.GetBytes(), TKmer::NBYTES);
                    memcpy(addrs2fill + TKmer::NBYTES, &readid, sizeof(ReadId));
                    memcpy(addrs2fill + TKmer::NBYTES + sizeof(ReadId), &pos, sizeof(PosInRead));
                    addrs2fill += seedbytes;
                }
                kmerseeds[i * nthreads + j].clear();
            }
            kmerseeds[i].clear();
        }
        timer.stop_and_log("K-mer packing");
        timer.start();

        std::vector<uint8_t> recvbuf(total_buf, 0);
        // MPI_ALLTOALLV(sendbuf.data(), sendcnt.data(), sdispls.data(), MPI_BYTE, recvbuf.data(), recvcnt.data(), rdispls.data(), MPI_BYTE, commgrid->GetWorld());
        MPI_ALLTOALL(sendbuf.data(), buf_size, MPI_BYTE, recvbuf.data(), buf_size, MPI_BYTE, commgrid->GetWorld());
        // compared to the MPI_ALLTOALLV, MPI_ALLTOALL is faster when we have more data to send 
        // we need to have a trade off here 
        timer.stop_and_log("K-mer exchange");
        timer.start();

        size_t numkmerseeds = std::accumulate(rthrcnt.begin(), rthrcnt.end(), 0) / seedbytes;

        #if LOG_LEVEL >= 2
        logger() << "received a total of " << numkmerseeds << " valid 'row' k-mers in second ALLTOALL exchange";
        logger.Flush("K-mers received:");
        #endif

        uint8_t *addrs2read = recvbuf.data();
        std::vector<std::vector<KmerSeedStruct>> recv_kmerseeds(nthreads);

        // calculate how many kmer seeds each thread will receive
        for (int i = 0; i < nthreads; i++) {
            for (int j = 0; j < nprocs; j++) {
                numkmerseeds_thread[i] += (rthrcnt[j * nthreads + i] / seedbytes);
            }
        }

        assert(numkmerseeds == std::accumulate(numkmerseeds_thread.begin(), numkmerseeds_thread.end(), 0));

        for (int i = 0; i < nthreads; i++) {
            recv_kmerseeds[i].reserve(numkmerseeds_thread[i]);
        }

        for (size_t i = 0; i < nprocs; i++)
        {
            addrs2read = recvbuf.data() + i * buf_size;
            for (size_t j = 0; j < nthreads; j++){
                for (size_t k = 0; k < rthrcnt[ i * nthreads + j ]; k++ ){
                    ReadId readid = *((ReadId*)(addrs2read + TKmer::NBYTES));
                    PosInRead pos = *((PosInRead*)(addrs2read + TKmer::NBYTES + sizeof(ReadId)));
                    TKmer kmer((void*)addrs2read);
                    recv_kmerseeds[j].push_back(KmerSeedStruct(kmer, readid, pos));
                    addrs2read += seedbytes;
                }
            }
            
        }

        timer.stop_and_log("K-mer stored into local data structure");
    }
    
   
    timer.start();

    // std::sort(recv_kmerseeds.data(), recv_kmerseeds.data() + numkmerseeds);
    omp_set_num_threads(nthreads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        assert(numkmerseeds_thread[tid] == recv_kmerseeds[tid].size());
        paradis::sort<KmerSeedStruct, TKmer::NBYTES>(recv_kmerseeds[tid].data(), recv_kmerseeds[tid].data() + numkmerseeds_thread[tid], per_thread);

    }

    // yfli: this barrier can be removed
    timer.stop_and_log("Shared memory parallel K-mer sorting");

    for(int tid = 0; tid < nthreads; tid++)
            logger() << "thread id: " << tid << " is responsible for"<< recv_kmerseeds[tid].size()<<std::endl;
    logger.Flush("cnt");

    timer.start();

    uint64_t valid_kmer[nthreads];
    KmerList kmerlists[nthreads];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t cur_kmer_cnt = 1;

        kmerlists[tid].reserve(int(numkmerseeds_thread[tid] / LOWER_KMER_FREQ));    // This should be enough
        TKmer last_mer = recv_kmerseeds[tid][0].kmer;
        valid_kmer[tid] = 0;

        for(size_t idx = 1; idx < numkmerseeds_thread[tid]; idx++) {
            TKmer cur_mer = recv_kmerseeds[tid][idx].kmer;
            if (cur_mer == last_mer) {
                cur_kmer_cnt++;
            } else {
                if (cur_kmer_cnt >= LOWER_KMER_FREQ && cur_kmer_cnt <= UPPER_KMER_FREQ) {

                    kmerlists[tid].push_back(KmerListEntry());
                    KmerListEntry& entry    = kmerlists[tid].back();

                    TKmer& kmer             = std::get<0>(entry);
                    READIDS& readids        = std::get<1>(entry);
                    POSITIONS& positions    = std::get<2>(entry);
                    int& count              = std::get<3>(entry);

                    count = cur_kmer_cnt;
                    kmer = last_mer;

                    for (size_t j = idx - count; j < idx; j++) {
                        readids[j - idx + count] = recv_kmerseeds[tid][j].readid;
                        positions[j - idx + count] = recv_kmerseeds[tid][j].posinread;
                    }

                    valid_kmer[tid]++;

                }
                cur_kmer_cnt = 1;
                last_mer = cur_mer;
            }
        }
        // deal with the last kmer
        if (cur_kmer_cnt >= LOWER_KMER_FREQ && cur_kmer_cnt <= UPPER_KMER_FREQ) {
            kmerlists[tid].push_back(KmerListEntry());
            KmerListEntry& entry         = kmerlists[tid].back();

            TKmer& kmer             = std::get<0>(entry);
            READIDS& readids        = std::get<1>(entry);
            POSITIONS& positions    = std::get<2>(entry);
            int& count              = std::get<3>(entry);

            count = cur_kmer_cnt;
            kmer = last_mer;

            for (size_t j = numkmerseeds_thread[tid] - count; j < numkmerseeds_thread[tid]; j++) {
                readids[j - numkmerseeds_thread[tid] + count] = recv_kmerseeds[tid][j].readid;
                positions[j - numkmerseeds_thread[tid] + count] = recv_kmerseeds[tid][j].posinread;
            }

            valid_kmer[tid]++;
        }
    }

    for (int i = 0; i < nthreads; i++) {
        logger() << valid_kmer[i] << " ";
    }
    logger.Flush("valid_kmer");

    // yfli: actually we don't need to copy it into one single kmerlist

    uint64_t valid_kmer_total = std::accumulate(valid_kmer, valid_kmer + nthreads, 0);
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(valid_kmer_total);

    for (int i = 0; i < nthreads; i++) {
        kmerlist->insert(kmerlist->end(), kmerlists[i].begin(), kmerlists[i].end());
    }

    int count  = std::get<3>(kmerlist->front());

    logger() << "valid_kmer_total: " << valid_kmer_total << std::endl;
    logger.Flush("valid_kmer_total");
    timer.stop_and_log("K-mer filtering");

    return std::unique_ptr<KmerList>(kmerlist);
    
    // yfli: i'm not using hyperloglog currently, may lead to some waste of memory.
}

int GetKmerOwner(const TKmer& kmer, int nprocs)
{
    uint64_t myhash = kmer.GetHash();
    double range = static_cast<double>(myhash) * static_cast<double>(nprocs);
    size_t owner = range / std::numeric_limits<uint64_t>::max();
    assert(owner >= 0 && owner < static_cast<int>(nprocs));
    return static_cast<int>(owner);
}

std::unique_ptr<CT<PosInRead>::PSpParMat>
create_kmer_matrix(const DnaBuffer& myreads, const KmerList& kmerlist, std::shared_ptr<CommGrid> commgrid)
{
    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();

    int64_t kmerid = kmerlist.size();
    int64_t totkmers = kmerid;
    int64_t totreads = myreads.size();

    MPI_Allreduce(&kmerid,      &totkmers, 1, MPI_INT64_T, MPI_SUM, commgrid->GetWorld());
    MPI_Allreduce(MPI_IN_PLACE, &totreads, 1, MPI_INT64_T, MPI_SUM, commgrid->GetWorld());

    MPI_Exscan(MPI_IN_PLACE, &kmerid, 1, MPI_INT64_T, MPI_SUM, commgrid->GetWorld());
    if (myrank == 0) kmerid = 0;

    std::vector<int64_t> local_rowids, local_colids;
    std::vector<PosInRead> local_positions;

    for (size_t i = 0; i < kmerlist.size(); i++)
    {
        const READIDS& readids = std::get<1>(kmerlist[i]);
        const POSITIONS& positions = std::get<2>(kmerlist[i]);
        int cnt = std::get<3>(kmerlist[i]);

        for (int j = 0; j < cnt; ++j)
        {
            local_colids.push_back(kmerid);
            local_rowids.push_back(readids[j]);
            local_positions.push_back(positions[j]);
        }

        kmerid++;
    }

    CT<int64_t>::PDistVec drows(local_rowids, commgrid);
    CT<int64_t>::PDistVec dcols(local_colids, commgrid);
    CT<PosInRead>::PDistVec dvals(local_positions, commgrid);

    return std::make_unique<CT<PosInRead>::PSpParMat>(totreads, totkmers, drows, dcols, dvals, false);
}
