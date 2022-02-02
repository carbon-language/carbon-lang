//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Alloc>
// struct allocator_traits
// {
//     typedef Alloc::propagate_on_container_copy_assignment
//           | false_type                   propagate_on_container_copy_assignment;
//     ...
// };

#include <memory>
#include <type_traits>

#include "test_macros.h"

template <class T>
struct A
{
    typedef T value_type;
    typedef std::true_type propagate_on_container_copy_assignment;
};

template <class T>
struct B
{
    typedef T value_type;
};


template <class T>
struct C
{
    typedef T value_type;
private:
    typedef std::true_type propagate_on_container_copy_assignment;
};

int main(int, char**)
{
    static_assert((std::is_same<std::allocator_traits<A<char> >::propagate_on_container_copy_assignment, std::true_type>::value), "");
    static_assert((std::is_same<std::allocator_traits<B<char> >::propagate_on_container_copy_assignment, std::false_type>::value), "");
#if TEST_STD_VER >= 11
    static_assert((std::is_same<std::allocator_traits<C<char> >::propagate_on_container_copy_assignment, std::false_type>::value), "");
#endif

  return 0;
}
