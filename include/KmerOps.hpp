#ifndef ELBA_KMEROPS_HPP
#define ELBA_KMEROPS_HPP

#include "common.h"
#include "Kmer.hpp"
#include "DnaSeq.hpp"
#include "DnaBuffer.hpp"
#include "HyperLogLog.hpp"

#ifndef MAX_ALLTOALL_MEM
#define MAX_ALLTOALL_MEM (128ULL * 1024ULL * 1024ULL * 1024ULL)
#endif

typedef uint32_t PosInRead;
typedef  int64_t ReadId;

typedef std::array<PosInRead, UPPER_KMER_FREQ> POSITIONS;
typedef std::array<ReadId,    UPPER_KMER_FREQ> READIDS;

typedef std::tuple<TKmer, ReadId, PosInRead> KmerSeed;
typedef std::tuple<READIDS, POSITIONS, int> KmerCountEntry;
typedef std::unordered_map<TKmer, KmerCountEntry> KmerCountMap;

// yfli: just experimenting, this part of code definitely need some refactoring

struct KmerSeedStruct{
    uint64_t kmer;      // yfli: temporary solution, need to be refactored
    ReadId readid;
    PosInRead posinread;

    // KmerSeedStruct(TKmer kmer, ReadId readid, PosInRead posinread) : kmer(kmer), readid(readid), posinread(posinread) {};
    KmerSeedStruct(uint64_t* kmer_addr, ReadId readid, PosInRead posinread) : kmer(*kmer_addr), readid(readid), posinread(posinread) {};
    KmerSeedStruct(const KmerSeedStruct& o) : kmer(o.kmer), readid(o.readid), posinread(o.posinread) {};
    KmerSeedStruct() {};

    long long GetByte(int &i) const
    {
        return (kmer >> (8*i)) & 0xff;
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

// used in the radix sort approach
typedef std::tuple<TKmer, READIDS, POSITIONS, int> KmerListEntry;
typedef std::vector<KmerListEntry> KmerList;

std::unique_ptr<CT<PosInRead>::PSpParMat>
create_kmer_matrix(const DnaBuffer& myreads, const KmerList& kmerlist, std::shared_ptr<CommGrid> commgrid);

std::unique_ptr<KmerCountMap>
get_kmer_count_map_keys(const DnaBuffer& myreads, std::shared_ptr<CommGrid> commgrid);

std::unique_ptr<KmerList>
get_kmer_list(const DnaBuffer& myreads, std::shared_ptr<CommGrid> commgrid);

void get_kmer_count_map_values(const DnaBuffer& myreads, KmerCountMap& kmermap, std::shared_ptr<CommGrid> commgrid);
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
    int nprocs;
    std::vector<std::vector<TKmer>>& kmerbuckets;

    KmerPartitionHandler(std::vector<std::vector<TKmer>>& kmerbuckets) : nprocs(kmerbuckets.size()), kmerbuckets(kmerbuckets) {}

    void operator()(const TKmer& kmer, size_t kid, size_t rid)
    {
        kmerbuckets[GetKmerOwner(kmer, nprocs)].push_back(kmer);
    }

    void operator()(const TKmer& kmer, BatchState& state, size_t kid)
    {
        auto& kmerbucket = kmerbuckets[GetKmerOwner(kmer, nprocs)];
        kmerbucket.push_back(kmer);
        state.mymaxsending = std::max(kmerbucket.size(), state.mymaxsending);
    }
};

struct KmerParserHandler
{
    int nprocs;
    ReadId readoffset;
    std::vector<std::vector<KmerSeed>>& kmerseeds;

    KmerParserHandler(std::vector<std::vector<KmerSeed>>& kmerseeds, ReadId readoffset) : nprocs(kmerseeds.size()), readoffset(readoffset), kmerseeds(kmerseeds) {}

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
