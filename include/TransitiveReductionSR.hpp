// Created by Giulia Guidi on 09/04/20.

#ifndef __TER_DEFS_H__
#define __TER_DEFS_H__

#include "../include/kmer/CommonKmers.hpp"

#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>

#include <string>
#include <sstream>

using namespace std;

#define FUZZ (1000)
#define DEBUG

/** Given a biridrected graph, an edge v ?-? x can only be considered transitive given a pair of edges v ?-? w ?-? x if:
 * (1) The two heads adjacent to w have opposite orientation: 
 *      2nd bit != 1st bit in MinPlus semiring B = A^2 such as 01 and 01 or 10 and 10;
 * (2) The heads adjacent to v in v ?-? w and v ?-? x have the same orientation, and
 * (3) The heads adjacent to x in v ?-? x and w ?-? x have the same orientation:
 *      1st and 2nd bit in M == 1st and 2nd bit in B during I = M >= B.
*/

dibella::CommonKmers compose(dibella::CommonKmers& me, const uint& suffix, const ushort& dir)
{ 
    me.overhang = suffix << 2 | dir;
    return me; 
}

void tobinary(ushort n, int* arr) 
{ 
    int nbit = 2;
    for(int i = 0; i < nbit; i++)
    { 
        arr[i] = n % 2; 
        n = n / 2; 
    }
} 

bool testdir(ushort dir1, ushort dir2)
{
    ushort rbit, lbit;

    int mybin1[2] = {0, 0};
    int mybin2[2] = {0, 0};
    
    if(dir1 != 0) tobinary(dir1, mybin1);
    if(dir2 != 0) tobinary(dir2, mybin2);

    rbit = mybin1[0];
    lbit = mybin2[1];

    if(rbit != lbit) return true;
    else return false;
}

// uint length1(const dibella::CommonKmers& me) { return me.overhang[0] >> 2; }
// uint length2(const dibella::CommonKmers& me) { return me.overhang[1] >> 2; }

// ushort  dir1(const dibella::CommonKmers& me) { return me.overhang[0]  & 3; }
// ushort  dir2(const dibella::CommonKmers& me) { return me.overhang[1]  & 3; }

uint length(const dibella::CommonKmers& me) { return me.overhang >> 2; }
ushort  dir(const dibella::CommonKmers& me) { return me.overhang  & 3; }

dibella::CommonKmers min(const dibella::CommonKmers& arg1, const dibella::CommonKmers& arg2) {
    if(length(arg2) < length(arg1)) return arg2;
    else return arg1;
}

dibella::CommonKmers max(const dibella::CommonKmers& arg1, const dibella::CommonKmers& arg2) {
    if(length(arg2) > length(arg1)) return arg2;
    else return arg1;
}

const uint infplus(const dibella::CommonKmers& a, const dibella::CommonKmers& b) {
	uint inf = std::numeric_limits<uint>::max();
    if (length(a) == inf || length(b) == inf) {
    	return inf;
    }
    return length(a) + length(b);
}

// GGGG: best (minimum overhang) overlap semiring on vector
template <class T1, class T2, class OUT>
struct BestOverlapVSRing : binary_function <T1, T2, OUT>
{
    OUT operator() (const T1& x, const T2& y) const
    {
        if(length(y) < length(x)) return static_cast<OUT>(y);
        else return static_cast<OUT>(x);
    }
};

// GGGG: best (minimum overhang) overlap semiring on matrix
template <class T1, class T2, class OUT>
struct BestOverlapMSRing : binary_function <T1, T2, OUT>
{
    OUT operator() (T1& x, const T2& y) const
    {
        /* I want to only keep entry corresponding to the best overlap for that column */
        if(length(x) != length(y)) x.overhang = 0;
        return static_cast<OUT>(x);
    }
};

template <class T1, class T2, class OUT>
struct Bind2ndBiSRing : binary_function <T1, T2, OUT>
{
    OUT operator() (const T1& x, const T2& y) const
    {
        return static_cast<OUT>(y);
    }
};

template <class T1, class T2, class OUT>
struct ReduceMBiSRing : binary_function <T1, T2, OUT>
{
    OUT operator() (const T1& x, const T2& y) const
    {
        if(length(y) > length(x)) return static_cast<OUT>(y);
        else return static_cast<OUT>(x);
    }
};

template <class T, class OUT>
struct PlusFBiSRing : unary_function <T, OUT>
{
    OUT operator() (T& x) const
    {
        return static_cast<OUT>(compose(x, length(x) + FUZZ, dir(x)));
    }
};

template <class T1, class T2, class OUT>
struct MinPlusBiSRing
{
	static OUT id() 			{ return std::numeric_limits<OUT>::max(); };
	static bool returnedSAID() 	{ return false; 	}
	static MPI_Op mpi_op() 		{ return MPI_MIN; 	};

	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		return min(arg1, arg2);
	}
	static OUT multiply(const T1& arg1, const T2& arg2)
	{
        OUT res;
        // printf("dir1 %d dir2 %d\n", dir(arg1), dir(arg2));
        // printf("len1 %d len2 %d\n", length(arg1), length(arg2));
        if(testdir(dir(arg1), dir(arg2)))
        {
            uint len = infplus(arg1, arg2);
            return compose(res, len, dir(arg2));
        } 
        else 
        {
            return id();
        }
	}
	static void axpy(T1 a, const T2 & x, OUT & y)
	{
		y = min(y, multiply(a, x));
	}
};

template <class T1, class T2>
struct GreaterBinaryOp : binary_function <T1, T2, bool>
{
    bool operator() (const T1& x, const T2& y) const
    {
        if(length(x) >= length(y) && dir(y) == dir(x)) return true;
        else return false;
    }
};

template <class T1, class T2, class OUT>
struct MultiplyBinaryOp : binary_function <T1, T2, OUT>
{
    OUT operator() (const T1& x, const T2& y) const { return static_cast<OUT>(compose(length(x) * length(y), dir(x))); }
};

template <class T>
struct ZeroUnaryOp : unary_function <T, bool>
{
    bool operator() (const T& x) const { if(x == 0) return true; else return false; }
};

template <class T, class T2>
struct EWiseMulOp : binary_function <T, T2, T>
{
    T operator() (T& x, const T2& y) const
    {
        if(!y) x.overhang = 0;
        return x;
    }
};

template <class T>
struct ZeroOverhangSR : unary_function <T, bool>
{
    bool operator() (const T& x) const { if(x.overhang == 0) return true; else return false; }
};

#endif