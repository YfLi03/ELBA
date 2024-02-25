#include "PairwiseAlignment.hpp"
#include "DnaSeq.hpp"
#include "Logger.hpp"
#include <omp.h>

std::unique_ptr<CT<Overlap>::PSpParMat>
PairwiseAlignment(DistributedFastaData& dfd, CT<SharedSeeds>::PSpParMat& Bmat, int mat, int mis, int gap, int dropoff)
{
    FastaIndex& index = dfd.getindex();
    auto commgrid = index.getcommgrid();
    int myrank = commgrid->GetRank();
    int nprocs = commgrid->GetSize();
    MPI_Comm comm = commgrid->GetWorld();
    auto rowbuf = dfd.getrowbuf();
    auto colbuf = dfd.getcolbuf();

    size_t localnnzs = Bmat.seqptr()->getnnz(); /* number of k-mer seeds stored by my processor */
    std::vector<CT<SharedSeeds>::ref_tuples> alignseeds; /* the local k-mer seeds that we run alignments on are stored here as a vector of tuples */
    alignseeds.reserve(localnnzs);
    auto dcsc = Bmat.seqptr()->GetDCSC();

    int64_t rowoffset = dfd.getrowstartid();
    int64_t coloffset = dfd.getcolstartid();

    /*
     * Go through each local k-mer seed.
     */
    if (dcsc != nullptr)
        for (int64_t i = 0; i < dcsc->nzc; ++i)
            for (int64_t j = dcsc->cp[i]; j < dcsc->cp[i+1]; ++j)
            {
                int64_t localrow = dcsc->ir[j];
                int64_t localcol = dcsc->jc[i];
                int64_t globalrow = localrow + rowoffset;
                int64_t globalcol = localcol + coloffset;

                /*
                * We don't want to align the same pair of reads twice. Because B is symmetric,
                * we need to come up with a way of avoiding duplicate alignments while also
                * maintaining load balance (i.e. we don't want to just align nonzeros in
                * the global upper triangle because then almost half the processors won't have
                * anything to do).
                *
                * We do this as follows: if the nonzero is in the upper triangle of the the
                * local submatrix (but not on the local diagonal) then we want to align it.
                * If the the nonzero is on the local diagonal, but is in the upper triangle
                * of the global matrix, then we want to align it. This works because the local
                * upper triangles in the global lower triangle correspond one-to-one to the
                * local lower triangles in the global upper triangle. We let the the global
                * upper triangle handle the edge case of the diagonals.
                */

                if ((localrow < localcol) || (localrow <= localcol && globalrow < globalcol))
                {
                    alignseeds.emplace_back(localrow, localcol, &(dcsc->numx[j]));
                }
            }

    size_t nalignments = alignseeds.size();

    #if LOG_LEVEL >= 2
    size_t totalignments;
    MPI_ALLREDUCE(&nalignments, &totalignments, 1, MPI_SIZE_T, MPI_SUM, comm);
    Logger logger(commgrid);
    logger() << "performing " << nalignments << "/" << totalignments << " alignments";
    logger.Flush("Alignment Counts:");
    #endif

    /*
     * Construct the overlap graph matrix by running alignments.
     */

    std::vector<int64_t> local_rowids, local_colids;
    std::vector<Overlap> overlaps;

    overlaps.reserve(nalignments);

    std::vector<std::vector<int64_t>> t_local_rowids, t_local_colids;
    std::vector<std::vector<Overlap>> t_overlaps;

    int nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
    
    t_local_rowids.resize(nthreads);
    t_local_colids.resize(nthreads);
    t_overlaps.resize(nthreads);

    for ( size_t i = 0; i < nthreads; i++ )
    {
        t_local_rowids[i].reserve(nalignments / nthreads + 1);
        t_local_colids[i].reserve(nalignments / nthreads + 1);
        t_overlaps[i].reserve(nalignments / nthreads + 1);
    }

    #if LOG_LEVEL >= 2
    logger() << "using " << nthreads << "threads";
    logger.Flush("Thread Counts:");
    #endif

    #pragma omp parallel for
    for (size_t i = 0; i < nalignments; ++i)
    {
        int tid = omp_get_thread_num();

        int64_t localrow = std::get<0>(alignseeds[i]);
        int64_t localcol = std::get<1>(alignseeds[i]);

        const DnaSeq& seqQ = (*rowbuf)[localrow];
        const DnaSeq& seqT = (*colbuf)[localcol];

        PosInRead lenQ = seqQ.size();
        PosInRead lenT = seqT.size();

        std::tuple<PosInRead, PosInRead> len(lenQ, lenT);

        /* TODO: change the below two lines */
        t_overlaps[tid].emplace_back(len, std::get<2>(alignseeds[i])->getseeds()[0]);
        t_overlaps[tid].back().extend_overlap(seqQ, seqT, mat, mis, gap, dropoff);

        t_local_rowids[tid].push_back(localrow + rowoffset);
        t_local_colids[tid].push_back(localcol + coloffset);
    }

    // merge the vectors
    for ( size_t i = 0; i < nthreads; i++ )
    {
        local_rowids.insert(local_rowids.end(), t_local_rowids[i].begin(), t_local_rowids[i].end());
        local_colids.insert(local_colids.end(), t_local_colids[i].begin(), t_local_colids[i].end());
        overlaps.insert(overlaps.end(), t_overlaps[i].begin(), t_overlaps[i].end());
    }

    CT<int64_t>::PDistVec drows(local_rowids, commgrid);
    CT<int64_t>::PDistVec dcols(local_colids, commgrid);
    CT<Overlap>::PDistVec dvals(overlaps, commgrid);

    int64_t numreads = index.gettotrecords();

    auto R = std::make_unique<CT<Overlap>::PSpParMat>(numreads, numreads, drows, dcols, dvals, false);

    return std::move(R);
}
