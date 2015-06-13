//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Check that we don't allocate when trying to insert a duplicate value into a
// unordered_set. See PR12999 http://llvm.org/bugs/show_bug.cgi?id=12999

#include <cassert>
#include <unordered_set>
#include "count_new.hpp"
#include "MoveOnly.h"

int main()
{
    {
    std::unordered_set<int> s;
    assert(globalMemCounter.checkNewCalledEq(0));

    for(int i=0; i < 100; ++i)
        s.insert(3);

    assert(s.size() == 1);
    assert(s.count(3) == 1);
    assert(globalMemCounter.checkNewCalledEq(2));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    globalMemCounter.reset();
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    std::unordered_set<MoveOnly> s;
    assert(globalMemCounter.checkNewCalledEq(0));

    for(int i=0; i<100; i++)
        s.insert(MoveOnly(3));

    assert(s.size() == 1);
    assert(s.count(MoveOnly(3)) == 1);
    assert(globalMemCounter.checkNewCalledEq(2));
    }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    globalMemCounter.reset();
#endif
}
