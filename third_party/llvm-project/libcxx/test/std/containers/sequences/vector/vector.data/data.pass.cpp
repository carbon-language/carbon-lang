//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// pointer data();

#include <vector>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

struct Nasty {
    Nasty() : i_(0) {}
    Nasty(int i) : i_(i) {}
    ~Nasty() {}

    Nasty * operator&() const { assert(false); return nullptr; }
    int i_;
    };

int main(int, char**)
{
    {
        std::vector<int> v;
        assert(v.data() == 0);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int> v(100);
        assert(v.data() == std::addressof(v.front()));
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<Nasty> v(100);
        assert(v.data() == std::addressof(v.front()));
        assert(is_contiguous_container_asan_correct(v));
    }
#if TEST_STD_VER >= 11
    {
        std::vector<int, min_allocator<int>> v;
        assert(v.data() == 0);
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<int, min_allocator<int>> v(100);
        assert(v.data() == std::addressof(v.front()));
        assert(is_contiguous_container_asan_correct(v));
    }
    {
        std::vector<Nasty, min_allocator<Nasty>> v(100);
        assert(v.data() == std::addressof(v.front()));
        assert(is_contiguous_container_asan_correct(v));
    }
#endif

  return 0;
}
