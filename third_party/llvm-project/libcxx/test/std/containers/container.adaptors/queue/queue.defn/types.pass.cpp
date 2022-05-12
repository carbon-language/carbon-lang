//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class T, class Container = deque<T>>
// class queue
// {
// public:
//     typedef Container                                container_type;
//     typedef typename container_type::value_type      value_type;
//     typedef typename container_type::reference       reference;
//     typedef typename container_type::const_reference const_reference;
//     typedef typename container_type::size_type       size_type;
//
// protected:
//     container_type c;
// ...
// };

#include <queue>
#include <type_traits>

#include "test_macros.h"

struct test
    : private std::queue<int>
{
    test()
    {
        c.push_back(1);
    }
};

struct C
{
    typedef int value_type;
    typedef int& reference;
    typedef const int& const_reference;
    typedef int size_type;
};

int main(int, char**)
{
    static_assert(( std::is_same<std::queue<int>::container_type, std::deque<int> >::value), "");
    static_assert(( std::is_same<std::queue<int, std::vector<int> >::container_type, std::vector<int> >::value), "");
    static_assert(( std::is_same<std::queue<int, std::vector<int> >::value_type, int>::value), "");
    static_assert(( std::is_same<std::queue<int>::reference, std::deque<int>::reference>::value), "");
    static_assert(( std::is_same<std::queue<int>::const_reference, std::deque<int>::const_reference>::value), "");
    static_assert(( std::is_same<std::queue<int>::size_type, std::deque<int>::size_type>::value), "");
    static_assert(( std::uses_allocator<std::queue<int>, std::allocator<int> >::value), "");
    static_assert((!std::uses_allocator<std::queue<int, C>, std::allocator<int> >::value), "");
    test t;

  return 0;
}
