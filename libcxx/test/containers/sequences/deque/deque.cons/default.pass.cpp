//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// deque()

#include <deque>
#include <cassert>

#include "../../../stack_allocator.h"
#include "../../../NotConstructible.h"

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
}
