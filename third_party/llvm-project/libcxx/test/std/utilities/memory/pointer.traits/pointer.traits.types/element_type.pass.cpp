//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class Ptr>
// struct pointer_traits
// {
//     typedef <details> element_type;
//     ...
// };

#include <memory>
#include <type_traits>

#include "test_macros.h"

struct A
{
    typedef char element_type;
};

template <class T>
struct B
{
    typedef char element_type;
};

template <class T>
struct C
{
};

template <class T, class U>
struct D
{
};

template <class T, class U>
struct E
{
    static int element_type;
};

template <class T>
struct F {
private:
  typedef int element_type;
};

int main(int, char**)
{
    static_assert((std::is_same<std::pointer_traits<A>::element_type, char>::value), "");
    static_assert((std::is_same<std::pointer_traits<B<int> >::element_type, char>::value), "");
    static_assert((std::is_same<std::pointer_traits<C<int> >::element_type, int>::value), "");
    static_assert((std::is_same<std::pointer_traits<D<double, int> >::element_type, double>::value), "");
    static_assert((std::is_same<std::pointer_traits<E<double, int> >::element_type, double>::value), "");
#if TEST_STD_VER >= 11
    static_assert((std::is_same<std::pointer_traits<F<double>>::element_type, double>::value), "");
#endif


  return 0;
}
