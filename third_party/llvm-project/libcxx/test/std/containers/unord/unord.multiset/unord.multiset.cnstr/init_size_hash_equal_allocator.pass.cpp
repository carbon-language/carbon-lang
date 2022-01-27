//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_multiset

// unordered_multiset(initializer_list<value_type> il, size_type n,
//               const hasher& hf, const key_equal& eql, const allocator_type& a);

#include <unordered_set>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "test_allocator.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::unordered_multiset<int,
                                   test_hash<int>,
                                   test_equal_to<int>,
                                   test_allocator<int>
                                   > C;
        typedef int P;
        C c({
                P(1),
                P(2),
                P(3),
                P(4),
                P(1),
                P(2)
            },
            7,
            test_hash<int>(8),
            test_equal_to<int>(9),
            test_allocator<int>(10)
           );
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        assert(c.count(1) == 2);
        assert(c.count(2) == 2);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);
        assert(c.hash_function() == test_hash<int>(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() == test_allocator<int>(10));
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
    {
        typedef std::unordered_multiset<int,
                                   test_hash<int>,
                                   test_equal_to<int>,
                                   min_allocator<int>
                                   > C;
        typedef int P;
        C c({
                P(1),
                P(2),
                P(3),
                P(4),
                P(1),
                P(2)
            },
            7,
            test_hash<int>(8),
            test_equal_to<int>(9),
            min_allocator<int>()
           );
        LIBCPP_ASSERT(c.bucket_count() == 7);
        assert(c.size() == 6);
        assert(c.count(1) == 2);
        assert(c.count(2) == 2);
        assert(c.count(3) == 1);
        assert(c.count(4) == 1);
        assert(c.hash_function() == test_hash<int>(8));
        assert(c.key_eq() == test_equal_to<int>(9));
        assert(c.get_allocator() == min_allocator<int>());
        assert(!c.empty());
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(std::fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }

  return 0;
}
