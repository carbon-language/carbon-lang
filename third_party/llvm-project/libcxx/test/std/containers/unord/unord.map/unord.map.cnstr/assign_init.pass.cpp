//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// unordered_map& operator=(initializer_list<value_type> il);

#include <unordered_map>
#include <string>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>

#include "test_macros.h"
#include "../../../test_compare.h"
#include "../../../test_hash.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::allocator<std::pair<const int, std::string> > A;
        typedef std::unordered_map<int, std::string,
                                   test_hash<int>,
                                   test_equal_to<int>,
                                   A
                                   > C;
        typedef std::pair<int, std::string> P;
        C c =   {
                    P(4, "four"),
                    P(1, "four"),
                    P(2, "four"),
                };
        c =     {
                    P(1, "one"),
                    P(2, "two"),
                    P(3, "three"),
                    P(4, "four"),
                    P(1, "four"),
                    P(2, "four"),
                };
        assert(c.bucket_count() >= 5);
        assert(c.size() == 4);
        assert(c.at(1) == "one");
        assert(c.at(2) == "two");
        assert(c.at(3) == "three");
        assert(c.at(4) == "four");
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }
    {
        typedef min_allocator<std::pair<const int, std::string> > A;
        typedef std::unordered_map<int, std::string,
                                   test_hash<int>,
                                   test_equal_to<int>,
                                   A
                                   > C;
        typedef std::pair<int, std::string> P;
        C c =   {
                    P(4, "four"),
                    P(1, "four"),
                    P(2, "four"),
                };
        c =     {
                    P(1, "one"),
                    P(2, "two"),
                    P(3, "three"),
                    P(4, "four"),
                    P(1, "four"),
                    P(2, "four"),
                };
        assert(c.bucket_count() >= 5);
        assert(c.size() == 4);
        assert(c.at(1) == "one");
        assert(c.at(2) == "two");
        assert(c.at(3) == "three");
        assert(c.at(4) == "four");
        assert(static_cast<std::size_t>(std::distance(c.begin(), c.end())) == c.size());
        assert(static_cast<std::size_t>(std::distance(c.cbegin(), c.cend())) == c.size());
        assert(fabs(c.load_factor() - (float)c.size()/c.bucket_count()) < FLT_EPSILON);
        assert(c.max_load_factor() == 1);
    }

  return 0;
}
