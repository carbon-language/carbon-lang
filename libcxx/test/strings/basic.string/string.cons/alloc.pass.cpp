//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// explicit basic_string(const Allocator& a = Allocator());

#include <string>
#include <cassert>

#include "../test_allocator.h"
#include "../min_allocator.h"

template <class S>
void
test()
{
    {
    S s;
    assert(s.__invariants());
    assert(s.data());
    assert(s.size() == 0);
    assert(s.capacity() >= s.size());
    assert(s.get_allocator() == typename S::allocator_type());
    }
    {
    S s(typename S::allocator_type(5));
    assert(s.__invariants());
    assert(s.data());
    assert(s.size() == 0);
    assert(s.capacity() >= s.size());
    assert(s.get_allocator() == typename S::allocator_type(5));
    }
}

#if __cplusplus >= 201103L

template <class S>
void
test2()
{
    {
    S s;
    assert(s.__invariants());
    assert(s.data());
    assert(s.size() == 0);
    assert(s.capacity() >= s.size());
    assert(s.get_allocator() == typename S::allocator_type());
    }
    {
    S s(typename S::allocator_type{});
    assert(s.__invariants());
    assert(s.data());
    assert(s.size() == 0);
    assert(s.capacity() >= s.size());
    assert(s.get_allocator() == typename S::allocator_type());
    }
}

#endif

int main()
{
    test<std::basic_string<char, std::char_traits<char>, test_allocator<char> > >();
#if __cplusplus >= 201103L
    test2<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();
#endif
}
