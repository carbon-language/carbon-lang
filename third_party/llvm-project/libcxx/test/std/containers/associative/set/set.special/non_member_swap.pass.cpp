//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// void swap(set& m);

#include <set>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "../../../test_compare.h"

int main(int, char**)
{
    typedef int V;
    {
    typedef std::set<int> M;
    {
        M m1;
        M m2;
        M m1_save = m1;
        M m2_save = m2;
        swap(m1, m2);
        assert(m1 == m2_save);
        assert(m2 == m1_save);
    }
    {
        V ar2[] =
        {
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
        };
        M m1;
        M m2(ar2, ar2+sizeof(ar2)/sizeof(ar2[0]));
        M m1_save = m1;
        M m2_save = m2;
        swap(m1, m2);
        assert(m1 == m2_save);
        assert(m2 == m1_save);
    }
    {
        V ar1[] =
        {
            1,
            2,
            3,
            4
        };
        M m1(ar1, ar1+sizeof(ar1)/sizeof(ar1[0]));
        M m2;
        M m1_save = m1;
        M m2_save = m2;
        swap(m1, m2);
        assert(m1 == m2_save);
        assert(m2 == m1_save);
    }
    {
        V ar1[] =
        {
            1,
            2,
            3,
            4
        };
        V ar2[] =
        {
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
        };
        M m1(ar1, ar1+sizeof(ar1)/sizeof(ar1[0]));
        M m2(ar2, ar2+sizeof(ar2)/sizeof(ar2[0]));
        M m1_save = m1;
        M m2_save = m2;
        swap(m1, m2);
        assert(m1 == m2_save);
        assert(m2 == m1_save);
    }
    }
    {
        typedef test_allocator<V> A;
        typedef test_less<int> C;
        typedef std::set<int, C, A> M;
        V ar1[] =
        {
            1,
            2,
            3,
            4
        };
        V ar2[] =
        {
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
        };
        M m1(ar1, ar1+sizeof(ar1)/sizeof(ar1[0]), C(1), A(1, 1));
        M m2(ar2, ar2+sizeof(ar2)/sizeof(ar2[0]), C(2), A(1, 2));
        M m1_save = m1;
        M m2_save = m2;
        swap(m1, m2);
        assert(m1 == m2_save);
        assert(m2 == m1_save);
        assert(m1.key_comp() == C(2));
        assert(m1.get_allocator().get_id() == 1);
        assert(m2.key_comp() == C(1));
        assert(m2.get_allocator().get_id() == 2);
    }
    {
        typedef other_allocator<V> A;
        typedef test_less<int> C;
        typedef std::set<int, C, A> M;
        V ar1[] =
        {
            1,
            2,
            3,
            4
        };
        V ar2[] =
        {
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12
        };
        M m1(ar1, ar1+sizeof(ar1)/sizeof(ar1[0]), C(1), A(1));
        M m2(ar2, ar2+sizeof(ar2)/sizeof(ar2[0]), C(2), A(2));
        M m1_save = m1;
        M m2_save = m2;
        swap(m1, m2);
        assert(m1 == m2_save);
        assert(m2 == m1_save);
        assert(m1.key_comp() == C(2));
        assert(m1.get_allocator() == A(2));
        assert(m2.key_comp() == C(1));
        assert(m2.get_allocator() == A(1));
    }

  return 0;
}
