//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// forward_list(size_type n, const value_type& v, const allocator_type& a);

#include <forward_list>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef test_allocator<int> A;
        typedef A::value_type T;
        typedef std::forward_list<T, A> C;
        T v(6);
        unsigned N = 10;
        C c(N, v, A(12));
        unsigned n = 0;
        for (C::const_iterator i = c.begin(), e = c.end(); i != e; ++i, ++n)
            assert(*i == v);
        assert(n == N);
        assert(c.get_allocator() == A(12));
    }
#if TEST_STD_VER >= 11
    {
        typedef min_allocator<int> A;
        typedef A::value_type T;
        typedef std::forward_list<T, A> C;
        T v(6);
        unsigned N = 10;
        C c(N, v, A());
        unsigned n = 0;
        for (C::const_iterator i = c.begin(), e = c.end(); i != e; ++i, ++n)
            assert(*i == v);
        assert(n == N);
        assert(c.get_allocator() == A());
    }
#endif

  return 0;
}
