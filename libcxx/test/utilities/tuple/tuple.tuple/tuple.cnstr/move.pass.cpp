//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// tuple(tuple&& u);

#include <tuple>
#include <cassert>

#include "../MoveOnly.h"

struct ConstructsWithTupleLeaf
{
    ConstructsWithTupleLeaf() {}

    ConstructsWithTupleLeaf(ConstructsWithTupleLeaf const &) { assert(false); }
    ConstructsWithTupleLeaf(ConstructsWithTupleLeaf &&) {}

    template <class T>
    ConstructsWithTupleLeaf(T t)
    { assert(false); }
};

int main()
{
    {
        typedef std::tuple<> T;
        T t0;
        T t = std::move(t0);
    }
    {
        typedef std::tuple<MoveOnly> T;
        T t0(MoveOnly(0));
        T t = std::move(t0);
        assert(std::get<0>(t) == 0);
    }
    {
        typedef std::tuple<MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1));
        T t = std::move(t0);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
    }
    {
        typedef std::tuple<MoveOnly, MoveOnly, MoveOnly> T;
        T t0(MoveOnly(0), MoveOnly(1), MoveOnly(2));
        T t = std::move(t0);
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == 1);
        assert(std::get<2>(t) == 2);
    }
    // A bug in tuple caused __tuple_leaf to use its explicit converting constructor
    //  as its move constructor. This tests that ConstructsWithTupleLeaf is not called
    // (w/ __tuple_leaf)
    {
        typedef std::tuple<ConstructsWithTupleLeaf> d_t;
        d_t d((ConstructsWithTupleLeaf()));
        d_t d2(static_cast<d_t &&>(d));
    }
}
