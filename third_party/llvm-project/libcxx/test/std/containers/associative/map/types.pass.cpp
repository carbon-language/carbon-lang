//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// template <class Key, class T, class Compare = less<Key>,
//           class Allocator = allocator<pair<const Key, T>>>
// class map
// {
// public:
//     // types:
//     typedef Key                                      key_type;
//     typedef T                                        mapped_type;
//     typedef pair<const key_type, mapped_type>        value_type;
//     typedef Compare                                  key_compare;
//     typedef Allocator                                allocator_type;
//     typedef typename allocator_type::reference       reference;
//     typedef typename allocator_type::const_reference const_reference;
//     typedef typename allocator_type::pointer         pointer;
//     typedef typename allocator_type::const_pointer   const_pointer;
//     typedef typename allocator_type::size_type       size_type;
//     typedef typename allocator_type::difference_type difference_type;
//     ...
// };

#include <map>
#include <type_traits>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef std::map<int, double> C;
    static_assert((std::is_same<C::key_type, int>::value), "");
    static_assert((std::is_same<C::mapped_type, double>::value), "");
    static_assert((std::is_same<C::value_type, std::pair<const int, double> >::value), "");
    static_assert((std::is_same<C::key_compare, std::less<int> >::value), "");
    static_assert((std::is_same<C::allocator_type, std::allocator<std::pair<const int, double> > >::value), "");
    static_assert((std::is_same<C::reference, std::pair<const int, double>&>::value), "");
    static_assert((std::is_same<C::const_reference, const std::pair<const int, double>&>::value), "");
    static_assert((std::is_same<C::pointer, std::pair<const int, double>*>::value), "");
    static_assert((std::is_same<C::const_pointer, const std::pair<const int, double>*>::value), "");
    static_assert((std::is_same<C::size_type, std::size_t>::value), "");
    static_assert((std::is_same<C::difference_type, std::ptrdiff_t>::value), "");
    }
#if TEST_STD_VER >= 11
    {
    typedef std::map<int, double, std::less<int>, min_allocator<std::pair<const int, double>>> C;
    static_assert((std::is_same<C::key_type, int>::value), "");
    static_assert((std::is_same<C::mapped_type, double>::value), "");
    static_assert((std::is_same<C::value_type, std::pair<const int, double> >::value), "");
    static_assert((std::is_same<C::key_compare, std::less<int> >::value), "");
    static_assert((std::is_same<C::allocator_type, min_allocator<std::pair<const int, double> > >::value), "");
    static_assert((std::is_same<C::reference, std::pair<const int, double>&>::value), "");
    static_assert((std::is_same<C::const_reference, const std::pair<const int, double>&>::value), "");
    static_assert((std::is_same<C::pointer, min_pointer<std::pair<const int, double>>>::value), "");
    static_assert((std::is_same<C::const_pointer, min_pointer<const std::pair<const int, double>>>::value), "");
//  min_allocator doesn't have a size_type, so one gets synthesized
    static_assert((std::is_same<C::size_type, std::make_unsigned<C::difference_type>::type>::value), "");
    static_assert((std::is_same<C::difference_type, std::ptrdiff_t>::value), "");
    }
#endif

  return 0;
}
