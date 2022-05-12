//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_multimap

// iterator insert(const value_type& x);

#include <unordered_map>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template<class Container>
void do_insert_const_lvalue_test()
{
    typedef Container C;
    typedef typename C::iterator R;
    typedef typename C::value_type VT;
    C c;
    const VT v1(3.5, 3);
    R r = c.insert(v1);
    assert(c.size() == 1);
    assert(r->first == 3.5);
    assert(r->second == 3);

    const VT v2(3.5, 4);
    r = c.insert(v2);
    assert(c.size() == 2);
    assert(r->first == 3.5);
    assert(r->second == 4);

    const VT v3(4.5, 4);
    r = c.insert(v3);
    assert(c.size() == 3);
    assert(r->first == 4.5);
    assert(r->second == 4);

    const VT v4(5.5, 4);
    r = c.insert(v4);
    assert(c.size() == 4);
    assert(r->first == 5.5);
    assert(r->second == 4);
}

int main(int, char**)
{
    do_insert_const_lvalue_test<std::unordered_multimap<double, int> >();
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_multimap<double, int, std::hash<double>, std::equal_to<double>,
                            min_allocator<std::pair<const double, int>>> C;
        do_insert_const_lvalue_test<C>();
    }
#endif

  return 0;
}
