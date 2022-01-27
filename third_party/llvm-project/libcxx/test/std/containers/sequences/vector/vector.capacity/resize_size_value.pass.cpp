//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void resize(size_type sz, const value_type& x);

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "asan_testing.h"

int main(int, char**)
{
    {
        std::vector<int> v(100);
        v.resize(50, 1);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(v == std::vector<int>(50));
        v.resize(200, 1);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
        for (unsigned i = 0; i < 50; ++i)
            assert(v[i] == 0);
        for (unsigned i = 50; i < 200; ++i)
            assert(v[i] == 1);
    }
    {
        // Add 1 for implementations that dynamically allocate a container proxy.
        std::vector<int, limited_allocator<int, 300 + 1> > v(100);
        v.resize(50, 1);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200, 1);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
    }
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> v(100);
        v.resize(50, 1);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(is_contiguous_container_asan_correct(v));
        assert((v == std::vector<int, min_allocator<int>>(50)));
        v.resize(200, 1);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
        for (unsigned i = 0; i < 50; ++i)
            assert(v[i] == 0);
        for (unsigned i = 50; i < 200; ++i)
            assert(v[i] == 1);
    }
    {
        std::vector<int, min_allocator<int>> v(100);
        v.resize(50, 1);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        assert(is_contiguous_container_asan_correct(v));
        v.resize(200, 1);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
        assert(is_contiguous_container_asan_correct(v));
    }
#endif

  return 0;
}
