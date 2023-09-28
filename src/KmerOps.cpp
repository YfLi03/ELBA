
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

std::unique_ptr<KmerList>
get_kmer_list(const DnaBuffer& myreads, std::shared_ptr<CommGrid> commgrid)
{
    MPI_Comm comm = MPI_COMM_WORLD;

    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();
    Logger logger(commgrid);


    int thread_num = omp_get_max_threads();                         // yfli: is not working well
    logger() << thread_num << std::endl;
    logger.Flush("thread_num");


    size_t numreads = myreads.size();                              /* Number of locally stored reads */

    std::ostringstream rootlog;

    MPITimer timer(comm);

    timer.start();

    size_t readoffset = numreads;                                   // yfli: not sure if we need this initial value
    MPI_Exscan(&numreads, &readoffset, 1, MPI_SIZE_T, MPI_SUM, commgrid->GetWorld());
    if (!myrank) readoffset = 0;    

    std::vector<std::vector<KmerSeed>> kmerseeds(nprocs);

    /* prepare the vars for each thread */
    std::vector<std::vector<std::vector<KmerSeed>>> kmerseeds_vecs;
    std::vector<KmerParserHandler> parser_vecs;
    for (int i = 0; i < thread_num; i++) {
        kmerseeds_vecs.push_back(std::vector<std::vector<KmerSeed>>(nprocs));
    }

    for (int i = 0; i < thread_num; i++ ) {
        /* This is a little bit tricky, as we need to avoid the kmerseeds_vecs from reallocating */
        parser_vecs.push_back(KmerParserHandler(kmerseeds_vecs[i], static_cast<ReadId>(readoffset)));
    }

    /* merge the seed vecs into one */
    ForeachKmerParallel(myreads, parser_vecs, thread_num);
    for (int i = 0; i < thread_num; i++) {
        for (int j = 0; j < nprocs; j++) {
            kmerseeds[j].insert(kmerseeds[j].end(), kmerseeds_vecs[i][j].begin(), kmerseeds_vecs[i][j].end());
        }
        kmerseeds_vecs[i].clear();
    }

    timer.stop_and_log("Local K-mer parsing");
    timer.start();

    std::vector<MPI_Count_type> sendcnt(nprocs), recvcnt(nprocs);
    std::vector<MPI_Displ_type> sdispls(nprocs), rdispls(nprocs);

    constexpr size_t seedbytes = TKmer::NBYTES + sizeof(ReadId) + sizeof(PosInRead);

    #if LOG_LEVEL >= 2
    logger() << std::setprecision(4) << "sending 'row' k-mers to each processor in this amount (megabytes): {";
    #endif

    for (int i = 0; i < nprocs; ++i)
    {
        sendcnt[i] = kmerseeds[i].size() * seedbytes;

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
    MPI_ALLTOALL(sendcnt.data(), 1, MPI_COUNT_TYPE, recvcnt.data(), 1, MPI_COUNT_TYPE, commgrid->GetWorld());
    /*
    
    sdispls.front() = rdispls.front() = 0;
    std::partial_sum(sendcnt.begin(), sendcnt.end()-1, sdispls.begin()+1);
    std::partial_sum(recvcnt.begin(), recvcnt.end()-1, rdispls.begin()+1);

    size_t totsend = std::accumulate(sendcnt.begin(), sendcnt.end(), 0);
    size_t totrecv = std::accumulate(recvcnt.begin(), recvcnt.end(), 0);
    */
    // copying data to the sendbuf
    // yfli: maybe these memcpy can be combined into fewer operations by adjusting the data structure
    std::vector<uint8_t> sendbuf(total_buf, 0);

    # pragma omp parallel for num_threads(thread_num) schedule(static)
    for (int i = 0; i < nprocs; ++i)
    {
        assert(kmerseeds[i].size() == (sendcnt[i] / seedbytes));
        uint8_t *addrs2fill = sendbuf.data() + i * buf_size;
        for (MPI_Count_type j = 0; j < kmerseeds[i].size(); ++j)
        {
            auto& seeditr = kmerseeds[i][j];
            TKmer kmer = std::get<0>(seeditr);
            ReadId readid = std::get<1>(seeditr);
            PosInRead pos = std::get<2>(seeditr);
            memcpy(addrs2fill, kmer.GetBytes(), TKmer::NBYTES);
            memcpy(addrs2fill + TKmer::NBYTES, &readid, sizeof(ReadId));
            memcpy(addrs2fill + TKmer::NBYTES + sizeof(ReadId), &pos, sizeof(PosInRead));
            addrs2fill += seedbytes;
        }
        kmerseeds[i].clear();
    }
    timer.stop_and_log("K-mer packing");
    timer.start();

    std::vector<uint8_t> recvbuf(total_buf, 0);
    // MPI_ALLTOALLV(sendbuf.data(), sendcnt.data(), sdispls.data(), MPI_BYTE, recvbuf.data(), recvcnt.data(), rdispls.data(), MPI_BYTE, commgrid->GetWorld());
    MPI_ALLTOALL(sendbuf.data(), buf_size, MPI_BYTE, recvbuf.data(), buf_size, MPI_BYTE, commgrid->GetWorld());
    /* compared to the MPI_ALLTOALLV, MPI_ALLTOALL is faster when we have more data to send */
    /* we need to have a trade off here */
    timer.stop_and_log("K-mer exchange");
    timer.start();

    size_t numkmerseeds = std::accumulate(recvcnt.begin(), recvcnt.end(), 0) / seedbytes;

    #if LOG_LEVEL >= 2
    logger() << "received a total of " << numkmerseeds << " 'row' k-mers in second ALLTOALL exchange";
    logger.Flush("K-mers received:");
    #endif

    uint8_t *addrs2read = recvbuf.data();
    std::vector<KmerSeedStruct> recv_kmerseeds;
    recv_kmerseeds.reserve(numkmerseeds);

    for (size_t i = 0; i < nprocs; i++)
    {
        addrs2read = recvbuf.data() + i * buf_size;
        for (size_t j = 0; j < recvcnt[i] / seedbytes; j++)
        {
            ReadId readid = *((ReadId*)(addrs2read + TKmer::NBYTES));
            PosInRead pos = *((PosInRead*)(addrs2read + TKmer::NBYTES + sizeof(ReadId)));
            TKmer kmer((void*)addrs2read);
            recv_kmerseeds.push_back(KmerSeedStruct(kmer, readid, pos));

            addrs2read += seedbytes;
        }
    }

    timer.stop_and_log("K-mer stored into local data structure");
    timer.start();

    // std::sort(recv_kmerseeds.data(), recv_kmerseeds.data() + numkmerseeds);
    paradis::sort<KmerSeedStruct, TKmer::NBYTES>(recv_kmerseeds.data(), recv_kmerseeds.data() + numkmerseeds, thread_num);

    timer.stop_and_log("Shared memory parallel K-mer sorting");

    timer.start();

    size_t start_idx[thread_num + 1];
    
    // yfli: finding the starting idx for each thread
    // some bug will be introduced here, if the data size is too small
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        if ( tid == 0 ){
            start_idx[0] = 0;
        } else {
            size_t idx = tid * (numkmerseeds / nthreads);
            TKmer last_mer = recv_kmerseeds[idx].kmer;
            TKmer cur_mer = recv_kmerseeds[idx].kmer;

            while (true) {
                cur_mer = recv_kmerseeds[idx - 1].kmer;
                if(cur_mer != last_mer) {
                    start_idx[tid] = idx;
                    break;
                }
                idx--;
                last_mer = cur_mer;
            }
        }
    }

    start_idx[thread_num] = numkmerseeds;

    uint64_t valid_kmer[thread_num];
    KmerList kmerlists[thread_num];

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t cur_kmer_cnt = 1;

        kmerlists[tid].reserve(int(numkmerseeds / thread_num / LOWER_KMER_FREQ));    // This should be enough
        TKmer last_mer = recv_kmerseeds[start_idx[tid]].kmer;
        valid_kmer[tid] = 0;

        for(size_t cur_idx = start_idx[tid] + 1; cur_idx < start_idx[tid + 1]; cur_idx++) {
            TKmer cur_mer = recv_kmerseeds[cur_idx].kmer;
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

                    for (size_t j = cur_idx - count; j < cur_idx; j++) {
                        readids[j - cur_idx + count] = recv_kmerseeds[j].readid;
                        positions[j - cur_idx + count] = recv_kmerseeds[j].posinread;
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

            size_t cur_idx = start_idx[tid + 1];

            for (size_t j = cur_idx - count; j < cur_idx; j++) {
                readids[j - cur_idx + count] = recv_kmerseeds[j].readid;
                positions[j - cur_idx + count] = recv_kmerseeds[j].posinread;
            }

            valid_kmer[tid]++;
        }
    }

    for (int i = 0; i < thread_num; i++) {
        logger() << valid_kmer[i] << " ";
    }
    logger.Flush("valid_kmer");

    uint64_t valid_kmer_total = std::accumulate(valid_kmer, valid_kmer + thread_num, 0);
    KmerList* kmerlist = new KmerList();
    kmerlist->reserve(valid_kmer_total);

    for (int i = 0; i < thread_num; i++) {
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
