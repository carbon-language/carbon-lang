//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <deque>

// explicit deque(size_type n);

#include <deque>
#include <cassert>

#include "../../../stack_allocator.h"
#include "../../../DefaultOnly.h"

template <class T, class Allocator>
void
test(unsigned n)
{
    typedef std::deque<T, Allocator> C;
    typedef typename C::const_iterator const_iterator;
    assert(DefaultOnly::count == 0);
    {
    C d(n);
    assert(DefaultOnly::count == n);
    assert(d.size() == n);
    assert(distance(d.begin(), d.end()) == d.size());
#ifdef _LIBCPP_MOVE
    for (const_iterator i = d.begin(), e = d.end(); i != e; ++i)
        assert(*i == T());
#endif
    }
    assert(DefaultOnly::count == 0);
}

int main()
{
    test<DefaultOnly, std::allocator<DefaultOnly> >(0);
    test<DefaultOnly, std::allocator<DefaultOnly> >(1);
    test<DefaultOnly, std::allocator<DefaultOnly> >(10);
    test<DefaultOnly, std::allocator<DefaultOnly> >(1023);
    test<DefaultOnly, std::allocator<DefaultOnly> >(1024);
    test<DefaultOnly, std::allocator<DefaultOnly> >(1025);
    test<DefaultOnly, std::allocator<DefaultOnly> >(2047);
    test<DefaultOnly, std::allocator<DefaultOnly> >(2048);
    test<DefaultOnly, std::allocator<DefaultOnly> >(2049);
    test<DefaultOnly, std::allocator<DefaultOnly> >(4095);
    test<DefaultOnly, std::allocator<DefaultOnly> >(4096);
    test<DefaultOnly, std::allocator<DefaultOnly> >(4097);
    test<DefaultOnly, stack_allocator<DefaultOnly, 4096> >(4095);
}
