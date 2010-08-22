//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// priority_queue(priority_queue&& q);

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

#endif  // _LIBCPP_MOVE

int main()
{
#ifdef _LIBCPP_MOVE
    std::priority_queue<MoveOnly> qo(std::less<MoveOnly>(), make<std::vector<MoveOnly> >(5));
    std::priority_queue<MoveOnly> q = std::move(qo);
    assert(q.size() == 5);
    assert(q.top() == MoveOnly(4));
#endif  // _LIBCPP_MOVE
}
