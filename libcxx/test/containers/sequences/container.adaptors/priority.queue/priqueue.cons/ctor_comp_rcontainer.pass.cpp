//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <queue>

// explicit priority_queue(const Compare& comp, container_type&& c);

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
    std::priority_queue<MoveOnly> q(std::less<MoveOnly>(), make<std::vector<MoveOnly> >(5));
    assert(q.size() == 5);
    assert(q.top() == MoveOnly(4));
#endif
}
