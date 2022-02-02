//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>
// UNSUPPORTED: c++03, c++11, c++14

// template <class InputIterator, class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//    forward_list(InputIterator, InputIterator, Allocator = Allocator())
//    -> forward_list<typename iterator_traits<InputIterator>::value_type, Allocator>;
//

#include <forward_list>
#include <iterator>
#include <cassert>
#include <cstddef>
#include <climits> // INT_MAX

#include "deduction_guides_sfinae_checks.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "test_allocator.h"

struct A {};

int main(int, char**)
{

//  Test the explicit deduction guides
    {
    const int arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::forward_list fwl(std::begin(arr), std::end(arr));

    static_assert(std::is_same_v<decltype(fwl), std::forward_list<int>>, "");
    assert(std::equal(fwl.begin(), fwl.end(), std::begin(arr), std::end(arr)));
    }

    {
    const long arr[] = {INT_MAX, 1L, 2L, 3L };
    std::forward_list fwl(std::begin(arr), std::end(arr), std::allocator<long>());
    static_assert(std::is_same_v<decltype(fwl)::value_type, long>, "");
    assert(std::distance(fwl.begin(), fwl.end()) == 4); // no size for forward_list
    auto it = fwl.begin();
    assert(*it++ == INT_MAX);
    assert(*it++ == 1L);
    assert(*it++ == 2L);
    }

//  Test the implicit deduction guides

    {
//  We don't expect this one to work.
//  std::forward_list fwl(std::allocator<int>()); // deque (allocator &)
    }

    {
    std::forward_list fwl(1, A{}); // deque (size_type, T)
    static_assert(std::is_same_v<decltype(fwl)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(fwl)::allocator_type, std::allocator<A>>, "");
    assert(std::distance(fwl.begin(), fwl.end()) == 1); // no size for forward_list
    }

    {
    std::forward_list fwl(1, A{}, test_allocator<A>()); // deque (size_type, T, allocator)
    static_assert(std::is_same_v<decltype(fwl)::value_type, A>, "");
    static_assert(std::is_same_v<decltype(fwl)::allocator_type, test_allocator<A>>, "");
    assert(std::distance(fwl.begin(), fwl.end()) == 1); // no size for forward_list
    }

    {
    std::forward_list fwl{1U, 2U, 3U, 4U, 5U}; // deque(initializer-list)
    static_assert(std::is_same_v<decltype(fwl)::value_type, unsigned>, "");
    assert(std::distance(fwl.begin(), fwl.end()) == 5); // no size for forward_list
    auto it = fwl.begin();
    std::advance(it, 2);
    assert(*it == 3U);
    }

    {
    std::forward_list fwl({1.0, 2.0, 3.0, 4.0}, test_allocator<double>()); // deque(initializer-list, allocator)
    static_assert(std::is_same_v<decltype(fwl)::value_type, double>, "");
    static_assert(std::is_same_v<decltype(fwl)::allocator_type, test_allocator<double>>, "");
    assert(std::distance(fwl.begin(), fwl.end()) == 4); // no size for forward_list
    auto it = fwl.begin();
    std::advance(it, 3);
    assert(*it == 4.0);
    }

    {
    std::forward_list<long double> source;
    std::forward_list fwl(source); // deque(deque &)
    static_assert(std::is_same_v<decltype(fwl)::value_type, long double>, "");
    static_assert(std::is_same_v<decltype(fwl)::allocator_type, std::allocator<long double>>, "");
    assert(std::distance(fwl.begin(), fwl.end()) == 0); // no size for forward_list
    }

    {
        typedef test_allocator<short> Alloc;
        typedef test_allocator<int> ConvertibleToAlloc;

        {
        std::forward_list<short, Alloc> source;
        std::forward_list fwl(source, Alloc(2));
        static_assert(std::is_same_v<decltype(fwl), decltype(source)>);
        }

        {
        std::forward_list<short, Alloc> source;
        std::forward_list fwl(source, ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(fwl), decltype(source)>);
        }

        {
        std::forward_list<short, Alloc> source;
        std::forward_list fwl(std::move(source), Alloc(2));
        static_assert(std::is_same_v<decltype(fwl), decltype(source)>);
        }

        {
        std::forward_list<short, Alloc> source;
        std::forward_list fwl(std::move(source), ConvertibleToAlloc(2));
        static_assert(std::is_same_v<decltype(fwl), decltype(source)>);
        }
    }

    SequenceContainerDeductionGuidesSfinaeAway<std::forward_list, std::forward_list<int>>();

    return 0;
}
