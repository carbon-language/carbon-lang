//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// deque()

#include <deque>
#include <cassert>

#include "../../../stack_allocator.h"
#include "../../../NotConstructible.h"
#include "../../../min_allocator.h"

template <class T, class Allocator>
void
test()
{
    std::deque<T, Allocator> d;
    assert(d.size() == 0);
}

int main()
{
    test<int, std::allocator<int> >();
    test<NotConstructible, stack_allocator<NotConstructible, 1> >();
#if __cplusplus >= 201103L
    test<int, min_allocator<int> >();
    test<NotConstructible, min_allocator<NotConstructible> >();
#endif
}
