#ifndef LBL_DAL_DISTRIBUTEDFASTADATA_H
#define LBL_DAL_DISTRIBUTEDFASTADATA_H

#include "FastaData.hpp"
#include "Logger.hpp"

class DistributedFastaData
{
public:
    DistributedFastaData(FIndex index);

    size_t getrowstartid() const { return rowstartid; }
    size_t getcolstartid() const { return colstartid; }
    size_t getnumrowreads() const { return numrowreads; }
    size_t getnumcolreads() const { return numcolreads; }

private:
    FIndex index;
    size_t rowstartid, colstartid;
    size_t numrowreads, numcolreads;
};

// static size_t findnbrs(std::vector<NbrData>& neighbors, const std::vector<size_t>& idoffsets, size_t startid, size_t totreads, int rc_flag, Grid commgrid)
// {
    // int myrank = commgrid->GetRank();
    // size_t numadded = 0;

    // auto iditr = std::upper_bound(idoffsets.cbegin(), idoffsets.cend(), startid);
    // iditr--;

    // assert(*iditr <= startid);

    // while (*iditr < startid + totreads)
    // {
        // int reqrank = iditr - idoffsets.cbegin();
        // size_t reqstart = std::max(*iditr++, startid);
        // size_t reqend = std::min(*iditr, startid + totreads);
        // neighbors.emplace_back(reqstart, reqend-reqstart, reqrank, rc_flag);
        // numadded++;
    // }

    // return numadded;
// }

#endif //LBL_DAL_DISTRIBUTEDFASTADATA_H
