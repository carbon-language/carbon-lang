//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// iterator insert(const_iterator p, const value_type& x);

#include <unordered_set>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template<class Container>
void do_insert_hint_const_lvalue_test()
{
    typedef Container C;
    typedef typename C::iterator R;
    typedef typename C::value_type VT;
    C c;
    typename C::const_iterator e = c.end();
    const VT v1(3.5);
    R r = c.insert(e, v1);
    assert(c.size() == 1);
    assert(*r == 3.5);

    r = c.insert(r, v1);
    assert(c.size() == 1);
    assert(*r == 3.5);

    const VT v2(4.5);
    r = c.insert(c.end(), v2);
    assert(c.size() == 2);
    assert(*r == 4.5);

    const VT v3(5.5);
    r = c.insert(c.end(), v3);
    assert(c.size() == 3);
    assert(*r == 5.5);
}

int main(int, char**)
{
    do_insert_hint_const_lvalue_test<std::unordered_set<double> >();
#if TEST_STD_VER >= 11
    {
        typedef std::unordered_set<double, std::hash<double>,
                                std::equal_to<double>, min_allocator<double>> C;
        do_insert_hint_const_lvalue_test<C>();
    }
#endif

    return 0;
}
