//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03 && !stdlib=libc++

// <vector>

// iterator insert(const_iterator position, value_type&& x);

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "MoveOnly.h"
#include "min_allocator.h"
#include "asan_testing.h"

int main(int, char**)
{
    {
        std::vector<MoveOnly> v(100);
        std::vector<MoveOnly>::iterator i = v.insert(v.cbegin() + 10, MoveOnly(3));
        assert(v.size() == 101);
        assert(is_contiguous_container_asan_correct(v));
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == MoveOnly());
        assert(v[j] == MoveOnly(3));
        for (++j; j < 101; ++j)
            assert(v[j] == MoveOnly());
    }
    {
        std::vector<MoveOnly, limited_allocator<MoveOnly, 300> > v(100);
        std::vector<MoveOnly, limited_allocator<MoveOnly, 300> >::iterator i = v.insert(v.cbegin() + 10, MoveOnly(3));
        assert(v.size() == 101);
        assert(is_contiguous_container_asan_correct(v));
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == MoveOnly());
        assert(v[j] == MoveOnly(3));
        for (++j; j < 101; ++j)
            assert(v[j] == MoveOnly());
    }
    {
        std::vector<MoveOnly, min_allocator<MoveOnly> > v(100);
        std::vector<MoveOnly, min_allocator<MoveOnly> >::iterator i = v.insert(v.cbegin() + 10, MoveOnly(3));
        assert(v.size() == 101);
        assert(is_contiguous_container_asan_correct(v));
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == MoveOnly());
        assert(v[j] == MoveOnly(3));
        for (++j; j < 101; ++j)
            assert(v[j] == MoveOnly());
    }

  return 0;
}
