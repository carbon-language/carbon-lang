//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// void push(value_type&& v);

#include <queue>
#include <cassert>

#include "../../../../MoveOnly.h"

int main()
{
#ifdef _LIBCPP_MOVE
    std::queue<MoveOnly> q;
    q.push(MoveOnly(1));
    assert(q.size() == 1);
    assert(q.front() == MoveOnly(1));
    assert(q.back() == MoveOnly(1));
    q.push(MoveOnly(2));
    assert(q.size() == 2);
    assert(q.front() == MoveOnly(1));
    assert(q.back() == MoveOnly(2));
    q.push(MoveOnly(3));
    assert(q.size() == 3);
    assert(q.front() == MoveOnly(1));
    assert(q.back() == MoveOnly(3));
#endif  // _LIBCPP_MOVE
}
