
#include "KmerOps.hpp"
#include "Bloom.hpp"
#include "Logger.hpp"
#include "DnaSeq.hpp"
#include <cstring>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>

#if USE_BLOOM == 1
static Bloom *bm = nullptr;
#else
static_assert(USE_BLOOM == 0);
#endif

std::unique_ptr<KmerCountMap>
get_kmer_count_map_keys(const DnaBuffer& myreads, std::shared_ptr<CommGrid> commgrid)
{
    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();

    KmerCountMap *kmermap;                                         /* Received k-mers will be stored in this local hash table */
    HyperLogLog hll;                                               /* HyperLogLog counter initialized with 12 bits as default */
    size_t numreads;                                               /* Number of locally stored reads */
    size_t avgcardinality;                                         /* Average estimate for number of distinct k-mers per procesor (via Hyperloglog)*/
    double cardinality;                                            /* Total estimate for number of distinct k-mers in dataset (via Hyperloglog) */
    double mycardinality;                                          /* Local estimate for number of distinct k-mers originating on my processor (via Hyperloglog) */
    std::vector<std::vector<TKmer>> kmerbuckets(nprocs);           /* My processor's outgoing k-mer seed buckets, one for each destination processor */
    std::vector<MPI_Count_type> sendcnt(nprocs), recvcnt(nprocs);  /* My processor's ALLTOALL send and receive counts for phase one k-mer exchange */
    std::vector<MPI_Displ_type> sdispls(nprocs), rdispls(nprocs);  /* My processor's ALLTOALL send and receive displacements */
    std::vector<uint8_t> sendbuf, recvbuf;                         /* My processor's ALLTOALL send and receive buffers of k-mers (packed) */
    size_t totsend, totrecv;                                       /* My processor's total number of send and receive bytes */
    size_t numkmerseeds;                                           /* Total number of k-mer seeds received by my processor after unpacked receive buffer */
    Logger logger(commgrid);
    std::ostringstream rootlog;

    kmermap = new KmerCountMap;
    numreads = myreads.size();

    /*
     * Estimate the number of distinct k-mers in my local FASTA partition.
     */
    KmerEstimateHandler estimator(hll);
    ForeachKmer(myreads, estimator);
    mycardinality = hll.estimate();

    #if LOG_LEVEL >= 2
    logger() << std::setprecision(3) << std::fixed << mycardinality << " k-mers";
    logger.Flush("K-mer cardinality estimate");
    #endif

    /*
     * Estimate the number of distinct k-mers in the entire FASTA
     * by merging Hyperloglog counters from each processor.
     */
    hll.parallelmerge(commgrid->GetWorld());
    cardinality = hll.estimate();

    avgcardinality = static_cast<size_t>(std::ceil(cardinality / nprocs));

    #if LOG_LEVEL >= 2
    rootlog << "global 'column' k-mer cardinality (merging all " << nprocs << " procesors results) is " << std::setprecision(3) << std::fixed << cardinality << ", or an average of " << avgcardinality << " per processor" << std::endl;
    logger.Flush(rootlog, 0);
    #endif

    /*
     * Reserve memory for local hash table and Bloom filter using
     * distinct k-mer count estimates.
     */
    kmermap->reserve(avgcardinality);
    bm = new Bloom(static_cast<int64_t>(std::ceil(cardinality)), 0.05);

    BatchState batch_state(myreads.size(), commgrid);

    int batch_round = 1;

    size_t total_totsend = 0, total_totrecv = 0;

    do
    {
        ReadId curid = batch_state.myreadid;

        /*
         * Distribute k-mers parsed from local FASTA partition into
         * outgoing k-mer buckets (buckets are staging area for
         * packing ALLTOALL send buffer). Uses hash function to
         * determine destination processor, so that the number of
         * distinct k-mers sent to each processor is roughly equal.
         *
         * Note that the number of distinct k-mers not same as number of k-mers being sent.
         * The hash function attempts to balance the load of distinct k-mers
         * sent to each processor, not the number of individually packed k-mers
         * received by each processor. This is because the received k-mers are
         * immediately queried against a Bloom filter and hash table keyed
         * by the distinct k-mer.
         */
        KmerPartitionHandler partitioner(kmerbuckets);
        ForeachKmer(myreads, partitioner, batch_state);

        /*
         * ALLTOALL send counts: Number of k-mers my processor is sending to each other processor.
         */
        std::transform(kmerbuckets.cbegin(), kmerbuckets.cend(), sendcnt.begin(), [](const auto& bucket) { return bucket.size() * TKmer::NBYTES; });

        /*
         * ALLTOALL send displacements and total k-mers sending.
         */
        sdispls.front() = 0;
        std::partial_sum(sendcnt.begin(), sendcnt.end()-1, sdispls.begin()+1);
        totsend = sdispls.back() + sendcnt.back();

        /*
         * ALLTOALL receive counts: Number of k-mers my processor will receive from each other processor.
         */
        MPI_ALLTOALL(sendcnt.data(), 1, MPI_COUNT_TYPE, recvcnt.data(), 1, MPI_COUNT_TYPE, commgrid->GetWorld());

        /*
         * ALLTOALL receive displacements and total k-mers receiving.
         */
        rdispls.front() = 0;
        std::partial_sum(recvcnt.begin(), recvcnt.end()-1, rdispls.begin()+1);
        totrecv = rdispls.back() + recvcnt.back();

        /*
         * Allocate memory for send and receive buffers.
         */
        sendbuf.resize(totsend, 0);
        recvbuf.resize(totrecv, 0);

        /*
         * Pack outgoing k-mers into send buffer.
         */
        for (int i = 0; i < nprocs; ++i)
        {
            uint8_t *dest = sendbuf.data() + sdispls[i];

            for (MPI_Count_type j = 0; j < kmerbuckets[i].size(); ++j)
            {
                memcpy(dest, kmerbuckets[i][j].GetBytes(), TKmer::NBYTES);
                dest += TKmer::NBYTES;
            }

            kmerbuckets[i].clear();
        }

        /*
         * Communicate locally parsed k-mers to other processors in ALLTOALL fashion.
         */
        MPI_ALLTOALLV(sendbuf.data(), sendcnt.data(), sdispls.data(), MPI_BYTE, recvbuf.data(), recvcnt.data(), rdispls.data(), MPI_BYTE, commgrid->GetWorld());

        /*
         * Unpack incoming k-mers in a streaming fashion.
         */
        numkmerseeds = totrecv / TKmer::NBYTES;
        uint8_t *src = recvbuf.data();
        for (size_t i = 0; i < numkmerseeds; ++i)
        {
            TKmer mer(src);
            src += TKmer::NBYTES;

            /*
             * Check if incoming k-mer is already "inside" the local Bloom filter.
             */
            if (bm->Check(mer.GetBytes(), TKmer::NBYTES))
            {
                /*
                 * With high probability, the k-mer has already been
                 * inserted into the filter, and therefore is likely not a
                 * singleton k-mer. Insert it into local hash table partition
                 * if it hasn't already been.
                 */
                if (kmermap->find(mer) == kmermap->end())
                    kmermap->insert({mer, KmerCountEntry({}, {}, 0)});
            }
            else
            {
                /*
                 * k-mer definitely hasn't been seen before, therefore we
                 * add it to the Bloom filter. If this k-mer is a singleton,
                 * then we have effectively filtered it away from being
                 * queried on the local hash table.
                 */
                bm->Add(mer.GetBytes(), TKmer::NBYTES);
            }
        }

        size_t justsent = (totsend / TKmer::NBYTES);
        size_t justrecv = (totrecv / TKmer::NBYTES);

        total_totsend += justsent;
        total_totrecv += justrecv;

        #if LOG_LEVEL >= 2
        logger() << " sent " << justsent << " k-mers parsed from " << (batch_state.myreadid - curid) << " reads and received " << justrecv << " k-mers";
        rootlog << "Round " << batch_round++;
        logger.Flush(rootlog);
        #endif

    } while (!batch_state.Finished());

    return std::unique_ptr<KmerCountMap>(kmermap);
}

void get_kmer_count_map_values(const DnaBuffer& myreads, KmerCountMap& kmermap, std::shared_ptr<CommGrid> commgrid)
{
    Logger logger(commgrid);
    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();
    size_t numreads = myreads.size();
    std::vector<std::vector<KmerSeed>> kmerseeds(nprocs);
    size_t readoffset = numreads;

    MPI_Exscan(&numreads, &readoffset, 1, MPI_SIZE_T, MPI_SUM, commgrid->GetWorld());
    if (!myrank) readoffset = 0;

    KmerParserHandler parser(kmerseeds, static_cast<ReadId>(readoffset));
    ForeachKmer(myreads, parser);

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

    MPI_ALLTOALL(sendcnt.data(), 1, MPI_COUNT_TYPE, recvcnt.data(), 1, MPI_COUNT_TYPE, commgrid->GetWorld());

    sdispls.front() = rdispls.front() = 0;
    std::partial_sum(sendcnt.begin(), sendcnt.end()-1, sdispls.begin()+1);
    std::partial_sum(recvcnt.begin(), recvcnt.end()-1, rdispls.begin()+1);

    size_t totsend = std::accumulate(sendcnt.begin(), sendcnt.end(), 0);
    size_t totrecv = std::accumulate(recvcnt.begin(), recvcnt.end(), 0);

    std::vector<uint8_t> sendbuf(totsend, 0);

    for (int i = 0; i < nprocs; ++i)
    {
        assert(kmerseeds[i].size() == (sendcnt[i] / seedbytes));
        uint8_t *addrs2fill = sendbuf.data() + sdispls[i];
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

    std::vector<uint8_t> recvbuf(totrecv, 0);
    MPI_ALLTOALLV(sendbuf.data(), sendcnt.data(), sdispls.data(), MPI_BYTE, recvbuf.data(), recvcnt.data(), rdispls.data(), MPI_BYTE, commgrid->GetWorld());

    size_t numkmerseeds = totrecv / seedbytes;

    #if LOG_LEVEL >= 2
    logger() << "received a total of " << numkmerseeds << " 'row' k-mers in second ALLTOALL exchange";
    logger.Flush("K-mers received:");
    #endif

    uint8_t *addrs2read = recvbuf.data();

    for (size_t i = 0; i < numkmerseeds; ++i)
    {
        TKmer kmer(addrs2read);

        #if USE_BLOOM == 1
        if (!bm->Check(kmer.GetBytes(), TKmer::NBYTES))
            continue;

        #else
        static_assert(USE_BLOOM == 0);
        #endif

        ReadId readid = *((ReadId*)(addrs2read + TKmer::NBYTES));
        PosInRead pos = *((PosInRead*)(addrs2read + TKmer::NBYTES + sizeof(ReadId)));
        addrs2read += seedbytes;

        auto kmitr = kmermap.find(kmer);
        if (kmitr == kmermap.end()) continue;
        KmerCountEntry& entry = kmitr->second;

        READIDS& readids      = std::get<0>(entry);
        POSITIONS& positions  = std::get<1>(entry);
        int& count            = std::get<2>(entry);

        if (count >= UPPER_KMER_FREQ)
        {
            kmermap.erase(kmer);
            continue;
        }

        readids[count] = readid;
        positions[count] = pos;
        count++;
    }

    #if LOG_LEVEL >= 2
    logger() << numkmerseeds;

    #if USE_BLOOM == 1
    logger() << " row k-mers filtered by Bloom filter, hash table, and upper k-mer bound threshold into " << kmermap.size() << " semi-reliable 'column' k-mers";
    delete bm;

    #else
    static_assert(USE_BLOOM == 0);
    logger() << " row k-mers filtered by hash table and upper k-mer bound threshold into " << kmermap.size() << " semi-reliable 'column' k-mers";
    #endif

    logger.Flush("K-mer filtering:");
    #endif

    auto itr = kmermap.begin();
    while (itr != kmermap.end())
    {
        if (std::get<2>(itr->second) < LOWER_KMER_FREQ) itr = kmermap.erase(itr);
        else itr++;
    }

    #if LOG_LEVEL >= 2
    size_t numkmers = kmermap.size();

    MPI_Allreduce(MPI_IN_PLACE, &numkmers, 1, MPI_SIZE_T, MPI_SUM, commgrid->GetWorld());

    if (!myrank) std::cout << "A total of " << numkmers << " reliable 'column' k-mers found\n" << std::endl;
    MPI_Barrier(commgrid->GetWorld());
    #endif
}

int GetKmerOwner(const TKmer& kmer, int nprocs)
{
    uint64_t myhash = kmer.GetHash();
    double range = static_cast<double>(myhash) * static_cast<double>(nprocs);
    size_t owner = range / std::numeric_limits<uint64_t>::max();
    assert(owner >= 0 && owner < static_cast<int>(nprocs));
    return static_cast<int>(owner);
}
/*
void write_to_file(std::string output_dir, int myrank, const hysortk::KmerListS& kmerlist) {
    std::string filename = output_dir + "/kmerlist_" + std::to_string(myrank) + ".txt";
    std::ofstream out(filename);
    for (auto& itr : kmerlist) {
        out << itr.kmer << " " << itr.cnt << " ";
        for (int i = 0; i < itr.cnt; ++i) {
            out << itr.rid[i] << " " << itr.pos[i] << " ";
        }
        out << std::endl;
    }
    out.close();
}
*/

std::unique_ptr<CT<PosInRead>::PSpParMat>
create_kmer_matrix(const DnaBuffer& myreads, const hysortk::KmerListS& kmerlist, std::shared_ptr<CommGrid> commgrid)
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

    for (auto& itr : kmerlist)
    {
        auto& readids = itr.rid;
        auto& positions = itr.pos;
        int cnt = itr.cnt;

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
