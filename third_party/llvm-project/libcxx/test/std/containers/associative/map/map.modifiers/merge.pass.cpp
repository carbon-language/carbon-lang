//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <map>

// class map

// template <class C2>
//   void merge(map<key_type, value_type, C2, allocator_type>& source);
// template <class C2>
//   void merge(map<key_type, value_type, C2, allocator_type>&& source);
// template <class C2>
//   void merge(multimap<key_type, value_type, C2, allocator_type>& source);
// template <class C2>
//   void merge(multimap<key_type, value_type, C2, allocator_type>&& source);

#include <map>
#include <cassert>
#include "test_macros.h"
#include "Counter.h"

template <class Map>
bool map_equal(const Map& map, Map other)
{
    return map == other;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct throw_comparator
{
    bool& should_throw_;

    throw_comparator(bool& should_throw) : should_throw_(should_throw) {}

    template <class T>
    bool operator()(const T& lhs, const T& rhs) const
    {
        if (should_throw_)
            throw 0;
        return lhs < rhs;
    }
};
#endif

int main(int, char**)
{
    {
        std::map<int, int> src{{1, 0}, {3, 0}, {5, 0}};
        std::map<int, int> dst{{2, 0}, {4, 0}, {5, 0}};
        dst.merge(src);
        assert(map_equal(src, {{5,0}}));
        assert(map_equal(dst, {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}}));
    }

#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        bool do_throw = false;
        typedef std::map<Counter<int>, int, throw_comparator> map_type;
        map_type src({{1, 0}, {3, 0}, {5, 0}}, throw_comparator(do_throw));
        map_type dst({{2, 0}, {4, 0}, {5, 0}}, throw_comparator(do_throw));

        assert(Counter_base::gConstructed == 6);

        do_throw = true;
        try
        {
            dst.merge(src);
        }
        catch (int)
        {
            do_throw = false;
        }
        assert(!do_throw);
        assert(map_equal(src, map_type({{1, 0}, {3, 0}, {5, 0}}, throw_comparator(do_throw))));
        assert(map_equal(dst, map_type({{2, 0}, {4, 0}, {5, 0}}, throw_comparator(do_throw))));
    }
#endif
    assert(Counter_base::gConstructed == 0);
    struct comparator
    {
        comparator() = default;

        bool operator()(const Counter<int>& lhs, const Counter<int>& rhs) const
        {
            return lhs < rhs;
        }
    };
    {
        typedef std::map<Counter<int>, int, std::less<Counter<int>>> first_map_type;
        typedef std::map<Counter<int>, int, comparator> second_map_type;
        typedef std::multimap<Counter<int>, int, comparator> third_map_type;

        {
            first_map_type first{{1, 0}, {2, 0}, {3, 0}};
            second_map_type second{{2, 0}, {3, 0}, {4, 0}};
            third_map_type third{{1, 0}, {3, 0}};

            assert(Counter_base::gConstructed == 8);

            first.merge(second);
            first.merge(third);

            assert(map_equal(first, {{1, 0}, {2, 0}, {3, 0}, {4, 0}}));
            assert(map_equal(second, {{2, 0}, {3, 0}}));
            assert(map_equal(third, {{1, 0}, {3, 0}}));

            assert(Counter_base::gConstructed == 8);
        }
        assert(Counter_base::gConstructed == 0);
        {
            first_map_type first{{1, 0}, {2, 0}, {3, 0}};
            second_map_type second{{2, 0}, {3, 0}, {4, 0}};
            third_map_type third{{1, 0}, {3, 0}};

            assert(Counter_base::gConstructed == 8);

            first.merge(std::move(second));
            first.merge(std::move(third));

            assert(map_equal(first, {{1, 0}, {2, 0}, {3, 0}, {4, 0}}));
            assert(map_equal(second, {{2, 0}, {3, 0}}));
            assert(map_equal(third, {{1, 0}, {3, 0}}));

            assert(Counter_base::gConstructed == 8);
        }
        assert(Counter_base::gConstructed == 0);
    }
    assert(Counter_base::gConstructed == 0);
    {
        std::map<int, int> first;
        {
            std::map<int, int> second;
            first.merge(second);
            first.merge(std::move(second));
        }
        {
            std::multimap<int, int> second;
            first.merge(second);
            first.merge(std::move(second));
        }
    }
    return 0;
}
