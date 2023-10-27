#include "KmerOps.hpp"
#include "Logger.hpp"
#include "DnaSeq.hpp"
#include "MPITimer.hpp"
#include "ParadisSort.hpp"
#include "MemoryChecker.hpp"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include <omp.h>


std::unique_ptr<KmerSeedBuckets>
exchange_kmer(const DnaBuffer& myreads,
     std::shared_ptr<CommGrid> commgrid,
     int thr_per_task,
     int max_thr_membounded)
{
    MPI_Comm comm = MPI_COMM_WORLD;

    #if LOG_LEVEL >= 3
    MPITimer timer(comm);
    #endif

    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();
    bool single_node = (nprocs == 1);

    Logger logger(commgrid);
    std::ostringstream rootlog;

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

    #if LOG_LEVEL >= 3
    timer.start();
    #endif

    size_t readoffset = numreads;         // yfli: not sure if we need this initial value
    if (!single_node)
        MPI_Exscan(&numreads, &readoffset, 1, MPI_SIZE_T, MPI_SUM, commgrid->GetWorld());
    if (!myrank) readoffset = 0;    

    /* prepare the vars for each task */
    KmerSeedVecs kmerseeds_vecs;    
    for (int i = 0; i < nthr_membounded; i++ ) {
        kmerseeds_vecs.push_back(std::vector<std::vector<KmerSeedStruct>>(nprocs * ntasks));
    }

    std::vector<KmerParserHandler> parser_vecs;
    for (int i = 0; i < nthr_membounded; i++ ) {
        /* This is a little bit tricky, as we need to avoid the kmerseeds_vecs from reallocating */
        parser_vecs.push_back(KmerParserHandler(kmerseeds_vecs[i], static_cast<ReadId>(readoffset)));
    }

    #pragma omp parallel for num_threads(nthr_membounded)
    for (int i = 0; i < nthr_membounded; i++) {
        for(int j = 0; j < nprocs * ntasks; j++) {
            kmerseeds_vecs[i][j].reserve(size_t(1.1 * numreads / nprocs / ntasks / nthr_membounded));
        }
    }

    ForeachKmerParallel(myreads, parser_vecs, nthr_membounded);

    #if LOG_LEVEL >= 3
    timer.stop_and_log("K-mer partitioning");
    print_mem_log(nprocs, myrank, "After partitioning");
    timer.start();
    #endif

    KmerSeedBuckets* recv_kmerseeds = new KmerSeedBuckets(ntasks);

    if (nprocs == 1){

        #if LOG_LEVEL >= 3
        timer.start();
        #endif

        #pragma omp parallel for num_threads(nthr_membounded)
        for (int j = 0; j < nprocs * ntasks; j++) {
            for (int i = 0; i < nthr_membounded; i++) {
                (*recv_kmerseeds)[j].insert((*recv_kmerseeds)[j].end(), kmerseeds_vecs[i][j].begin(), kmerseeds_vecs[i][j].end());
            }
        }
        

        #if LOG_LEVEL >= 3
        timer.stop_and_log("Local K-mer format conversion (nprocs == 1) ");
        #endif

        /* for 1 process, no MPI communication is necessary*/
        return std::unique_ptr<KmerSeedBuckets>(recv_kmerseeds);
    } 

    /* more than 1 process, need MPI communication */

    std::vector<uint64_t> task_seedcnt(ntasks);                                 /* number of kmer seeds for each local task in total */
    std::vector<uint64_t> sthrcnt(nprocs * ntasks), rthrcnt(nprocs * ntasks);   /* number of kmer seeds for each global task (sending/reciving) */
    uint64_t batch_sz = std::numeric_limits<MPI_Count_type>::max() / nprocs;      
    batch_sz = std::min(batch_sz, (uint64_t)MAX_SEND_BATCH);

    constexpr size_t seedbytes = TKmer::NBYTES + sizeof(ReadId) + sizeof(PosInRead);

    #if LOG_LEVEL >= 2
    logger() << std::setprecision(4) << "sending 'row' k-mers to each processor in this amount : {";
    #endif

    for (int i = 0; i < nprocs; ++i)
    {
        for (int j = 0; j < ntasks; j++) 
        {
            for (int k = 0; k < nthr_membounded; k++)
            {
                sthrcnt[i * ntasks + j] += kmerseeds_vecs[k][i * ntasks + j].size();
            }

            #if LOG_LEVEL >= 2
            logger() << sthrcnt[i * ntasks + j]  << "+";
            #endif
        }

        #if LOG_LEVEL >= 2
        logger() << ", ";
        #endif
    }

    #if LOG_LEVEL >= 2
    logger() << "}";
    logger.Flush("K-mer exchange sendcounts:");
    #endif

    /* calculate the offset for each task */
    MPI_ALLTOALL(sthrcnt.data(), ntasks, MPI_UNSIGNED_LONG_LONG, rthrcnt.data(), ntasks, MPI_UNSIGNED_LONG_LONG, commgrid->GetWorld());

    /* calculate how many kmer seeds each thread will receive */
    size_t numkmerseeds = std::accumulate(rthrcnt.begin(), rthrcnt.end(), (uint64_t)0);

    for (int i = 0; i < ntasks; i++) {
        for (int j = 0; j < nprocs; j++) {
            task_seedcnt[i] += rthrcnt[j * ntasks + i];
        }
    }
    assert(numkmerseeds == std::accumulate(task_seedcnt.begin(), task_seedcnt.end(), (uint64_t)0));

    for (int i = 0; i < ntasks; i++) {
        (*recv_kmerseeds)[i].reserve(task_seedcnt[i]);
    }

    /* Sending in batches using Alltoall instead of Alltoallv */
    BatchSender bsender(commgrid, ntasks, nthr_membounded, batch_sz, kmerseeds_vecs);
    BatchStorer bstorer(*recv_kmerseeds, ntasks, nprocs, rthrcnt.data(), batch_sz);

    bsender.init();
    int round = 0;

    while(bsender.get_status() != BatchSender::Status::BATCH_DONE) {
        round += 1;

        uint8_t* recvbuf = bsender.progress();
        assert(recvbuf != nullptr);
        bstorer.store(recvbuf);
    }

    #if LOG_LEVEL >= 3
    std::ostringstream ss;
    ss << "K-mer exchange and storing (rounds: " << round << ", batch size: " << batch_sz << ")";
    timer.stop_and_log(ss.str().c_str());
    ss.clear();
    print_mem_log(nprocs, myrank, "After storing (not deleting buffers)");
    bsender.print_results(logger);
    #endif

    return std::unique_ptr<KmerSeedBuckets>(recv_kmerseeds);
}


std::unique_ptr<KmerList>
filter_kmer(std::unique_ptr<KmerSeedBuckets>& recv_kmerseeds, std::shared_ptr<CommGrid> commgrid, int thr_per_task )
{

    MPI_Comm comm = MPI_COMM_WORLD;
    MPITimer timer(comm);

    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();

    Logger logger(commgrid);
    std::ostringstream rootlog;

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

    #if LOG_LEVEL >= 2
    logger() <<"task num: "<< ntasks << " \tthread per task: " << thr_per_task <<" \ttask size: ";
    for (int i = 0; i < ntasks; i++) {
        task_seedcnt[i] = (*recv_kmerseeds)[i].size();
        logger() << task_seedcnt[i] << " ";
    }
    logger.Flush("Parallel tasks for kmer sorting and counting:");
    #endif

    #if LOG_LEVEL >= 3
    timer.start();
    #endif

    // yfli: maybe ask omp to scatter the threads on different places
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        paradis::sort<KmerSeedStruct, TKmer::NBYTES>((*recv_kmerseeds)[tid].data(), (*recv_kmerseeds)[tid].data() + task_seedcnt[tid], thr_per_task);
    

    #if LOG_LEVEL >= 3
    }
    /* this implicit barrier can be eliminated when not debugging */
    timer.stop_and_log("Shared memory parallel K-mer sorting");
    timer.start();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
    #endif


        kmerlists[tid].reserve(uint64_t(task_seedcnt[tid] / LOWER_KMER_FREQ));    // This should be enough
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

        /* deal with the last kmer */
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
    timer.stop_and_log("K-mer copying");
    #endif

    return std::unique_ptr<KmerList>(kmerlist);
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
