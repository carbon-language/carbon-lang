//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// queue(queue&& q);

#include <queue>
#include <cassert>

#include "../../../../MoveOnly.h"

#ifdef _LIBCPP_MOVE

template <class C>
C
make(int n)
{
    C c;
    for (int i = 0; i < n; ++i)
        c.push_back(MoveOnly(i));
    return c;
}

#endif

int main()
{
#ifdef _LIBCPP_MOVE
    std::queue<MoveOnly> q(make<std::deque<MoveOnly> >(5));
    std::queue<MoveOnly> q2 = std::move(q);
    assert(q2.size() == 5);
    assert(q.empty());
#endif
}
