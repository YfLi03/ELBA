#include "KmerOps.hpp"
#include "Logger.hpp"
#include "DnaSeq.hpp"
#include "MPITimer.hpp"
#include "ParadisSort.hpp"
#include "MemoryChecker.hpp"
#include "Supermer.hpp"
#include "compiletime.h"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <thread>
#include <deque>


ParallelData
prepare_supermer(const DnaBuffer& myreads,
     std::shared_ptr<CommGrid> comm,
     int thr_per_task,
     int max_thr_membounded)
{

    int myrank;
    int nprocs;
    myrank = comm->GetRank();
    nprocs = comm->GetSize();

    Logger logger(comm);

    /* parallel settings */
    omp_set_nested(1);
    int ntasks = omp_get_max_threads() / thr_per_task;
    if (ntasks < 1) {
        ntasks = 1;
        thr_per_task = omp_get_max_threads();
    }
    /* for memory bounded operations in this stage, we use another number of threads */
    int nthr_membounded = std::min(omp_get_max_threads() , max_thr_membounded);

    size_t numreads = myreads.size();     /* Number of locally stored reads */
    // !! There may be a bug with large file and small rank num. in this case myread can read things wrong, resulting in 
    // !! Kmers with all A and C.


#if LOG_LEVEL >= 2
    logger() << ntasks << " \t (thread per task: " << thr_per_task << ")";
    logger.Flush("Task num:");
    logger() << nthr_membounded ;
    logger.Flush("Thread count used for memory bounded operations:");
#endif

    size_t readoffset = numreads;      
    MPI_Exscan(&numreads, &readoffset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm->GetWorld());
      

    /* data structure for storing the data of different threads */
    ParallelData data(nprocs, ntasks, nthr_membounded, readoffset);

    /* find the destinations of each kmer */
    FindKmerDestinationsParallel(myreads, nthr_membounded, ntasks*nprocs, data);

#if LOG_LEVEL >= 3
    timer.stop_and_log("(Inc) K-mer destination finding");
    timer.start();
#endif

    /* encode the supermers */
    #pragma omp parallel num_threads(nthr_membounded)
    {
        int tid = omp_get_thread_num();
        auto& lengths = data.get_my_lengths(tid);
        auto& supermers = data.get_my_supermers(tid);
        auto& destinations = data.get_my_destinations(tid);
        auto& readids = data.get_my_readids(tid);

        SupermerEncoder encoder(lengths, supermers, readoffset, MAX_SUPERMER_LEN);

        for (size_t i = 0; i < readids.size(); ++i) {
            encoder.encode(destinations[i], myreads[readids[i]], readids[i] + readoffset);
        }

    }
    return data;
}



std::unique_ptr<KmerSeedBuckets> exchange_supermer(ParallelData& data, std::shared_ptr<CommGrid> comm)
{
    int myrank;
    int nprocs;
    myrank = comm->GetRank();
    nprocs = comm->GetSize();
    Logger logger(comm);

    int ntasks = data.ntasks;

    std::vector<size_t> send_counts(nprocs * ntasks, 0);
    std::vector<size_t> recv_counts(nprocs * ntasks, 0);

    for (int i = 0; i < nprocs; i++) {
        for (int j = 0; j < ntasks; j++) 
            send_counts[i * ntasks + j] = data.get_supermer_cnt(i, j);
    }

    MPI_Alltoall(send_counts.data(), ntasks, MPI_UNSIGNED_LONG_LONG, recv_counts.data(), ntasks, MPI_UNSIGNED_LONG_LONG, comm->GetWorld());

    std::vector<std::vector<uint32_t>> length;

    LengthExchanger length_exchanger(comm->GetWorld(), ntasks, MAX_SEND_BATCH, sizeof(uint32_t), data.nthr_membounded, data.lengths, recv_counts, length);
    length_exchanger.initialize();

    while(length_exchanger.status != LengthExchanger::Status::BATCH_DONE) {
        length_exchanger.progress();
    }

    KmerSeedBuckets* bucket = new KmerSeedBuckets(ntasks);
    SupermerExchanger supermer_exchanger(comm->GetWorld(), ntasks, MAX_SEND_BATCH, MAX_SUPERMER_LEN + sizeof(ReadId) + sizeof(PosInRead), data.nthr_membounded, data.lengths, data.supermers, recv_counts, length, *bucket); 
    
    supermer_exchanger.initialize();
    while (supermer_exchanger.status != SupermerExchanger::Status::BATCH_DONE)
    {
        supermer_exchanger.progress();
    }

#if LOG_LEVEL >= 3
timer.stop_and_log("(Inc) Supermer exchange");
supermer_exchanger.print_stats();
#endif
    

    return std::unique_ptr<KmerSeedBuckets>(bucket);
}



std::unique_ptr<KmerList>
filter_kmer(std::unique_ptr<KmerSeedBuckets>& recv_kmerseeds, 
     std::shared_ptr<CommGrid> comm,
     int thr_per_task)
{

    Logger logger(comm);
    int myrank;
    int nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    std::ostringstream rootlog;

    /* determine how many threads to use for each task */
    int ntasks = omp_get_max_threads() / thr_per_task;
    if (ntasks < 1){
        ntasks = 1;
        thr_per_task = omp_get_max_threads();
    }

    assert(ntasks == recv_kmerseeds->size());

    omp_set_nested(1);
    omp_set_num_threads(ntasks);

    uint64_t task_seedcnt[ntasks];
    uint64_t valid_kmer[ntasks];
    KmerList kmerlists[ntasks];


    for (int i = 0; i < ntasks; i++) {
        task_seedcnt[i] = (*recv_kmerseeds)[i].size();
    }
    

    // distribute available threads to each task in according to number of tasks
 //   int nthreads = omp_get_max_threads();
  //  int total_seeds = std::accumulate(task_seedcnt, task_seedcnt + ntasks, (uint64_t)0);


    /* sort the receiving vectors */
    #pragma omp parallel 
    {

        int tid = omp_get_thread_num();
//        int tasknum = std::max(1, (int)std::round((double)task_seedcnt[tid] * nthreads / total_seeds));
        paradis::sort<KmerSeedStruct, TKmer::NBYTES>((*recv_kmerseeds)[tid].data(), (*recv_kmerseeds)[tid].data() + task_seedcnt[tid], thr_per_task);


        /* do a linear scan and filter out the kmers we want */
        
        kmerlists[tid].reserve(uint64_t(task_seedcnt[tid] / LOWER_KMER_FREQ));
        TKmer last_mer = (*recv_kmerseeds)[tid][0].kmer;
        uint64_t cur_kmer_cnt = 1;
        valid_kmer[tid] = 0;

        for(size_t idx = 1; idx < task_seedcnt[tid]; idx++) {
            TKmer cur_mer = (*recv_kmerseeds)[tid][idx].kmer;
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
                        readids[j - idx + count] = (*recv_kmerseeds)[tid][j].readid;
                        positions[j - idx + count] = (*recv_kmerseeds)[tid][j].posinread;
                    }

                    valid_kmer[tid]++;
                }

                cur_kmer_cnt = 1;
                last_mer = cur_mer;
            }
        }

        /* deal with the last kmer of a task */
        if (cur_kmer_cnt >= LOWER_KMER_FREQ && cur_kmer_cnt <= UPPER_KMER_FREQ) {
            kmerlists[tid].push_back(KmerListEntry());
            KmerListEntry& entry         = kmerlists[tid].back();

            TKmer& kmer             = std::get<0>(entry);
            READIDS& readids        = std::get<1>(entry);
            POSITIONS& positions    = std::get<2>(entry);
            int& count              = std::get<3>(entry);

            count = cur_kmer_cnt;
            kmer = last_mer;

            for (size_t j = task_seedcnt[tid] - count; j < task_seedcnt[tid]; j++) {
                readids[j - task_seedcnt[tid] + count] = (*recv_kmerseeds)[tid][j].readid;
                positions[j - task_seedcnt[tid] + count] = (*recv_kmerseeds)[tid][j].posinread;
            }

            valid_kmer[tid]++;
        }
    }

#if LOG_LEVEL >= 3
    timer.stop_and_log("K-mer counting");
    timer.start();
#endif

#if LOG_LEVEL >= 2
    for (int i = 0; i < ntasks; i++) {
        logger() << valid_kmer[i] << " ";
    }
    logger.Flush("Valid kmer from tasks:");
#endif


    uint64_t valid_kmer_total = std::accumulate(valid_kmer, valid_kmer + ntasks, (uint64_t)0);
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(valid_kmer_total);

    // yfli: actually we don't need to copy it into one single kmerlist
    for (int i = 0; i < ntasks; i++) {
        kmerlist->insert(kmerlist->end(), kmerlists[i].begin(), kmerlists[i].end());
    }

    logger() << valid_kmer_total;
    logger.Flush("Valid kmer for process:");

#if LOG_LEVEL >= 3
    timer.stop_and_log("(Inc) K-mer copying");
#endif

    return std::unique_ptr<KmerList>(kmerlist);
}

void FindKmerDestinationsParallel(const DnaBuffer& myreads, int nthreads, int tot_tasks, ParallelData& data) {
    
    assert(nthreads > 0);

    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < myreads.size(); ++i)
    {
        int tid = omp_get_thread_num();

        auto &dest = data.register_new_destination(tid, i);
        if (myreads[i].size() < KMER_SIZE)
            continue;
        dest.reserve(myreads[i].size() - KMER_SIZE + 1);

        std::vector<TMmer> repmers = TMmer::GetRepMmers(myreads[i]);

        Minimizer_Deque deque;
        int head_pos = 0;

        /* initialize the deque */
        for(; head_pos < KMER_SIZE - MINIMIZER_SIZE; head_pos++) {
            deque.insert_minimizer(repmers[head_pos].GetHash(), head_pos);
        }
        int tail_pos = head_pos - KMER_SIZE + MINIMIZER_SIZE - 1;

        /* start the main loop */
        for(; head_pos < repmers.size(); head_pos++, tail_pos++) {
            deque.insert_minimizer(repmers[head_pos].GetHash(), head_pos);
            deque.remove_minimizer(tail_pos);
            dest.push_back(GetMinimizerOwner(deque.get_current_minimizer(), tot_tasks));
        }

    }
}


inline int GetMinimizerOwner(const uint64_t& hash, int tot_tasks) {
    // Need to check if this gives equal distribution
    return static_cast<int>(hash % tot_tasks);
}


inline int GetKmerOwner(const TKmer& kmer, int nprocs) {
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
