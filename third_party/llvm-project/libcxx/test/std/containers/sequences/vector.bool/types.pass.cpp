//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Test nested types and default template args:

// template <class Allocator>
// class vector<bool, Allocator
// {
// public:
//     typedef T                                        value_type;
//     typedef Allocator                                allocator_type;
//     typedef implementation-defined                   iterator;
//     typedef implementation-defined                   const_iterator;
//     typedef typename allocator_type::size_type       size_type;
//     typedef typename allocator_type::difference_type difference_type;
//     typedef typename allocator_type::pointer         pointer;
//     typedef typename allocator_type::const_pointer   const_pointer;
//     typedef std::reverse_iterator<iterator>          reverse_iterator;
//     typedef std::reverse_iterator<const_iterator>    const_reverse_iterator;
//     typedef bool                                     const_reference;
// };

#include <vector>
#include <iterator>
#include <type_traits>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../Copyable.h"
#include "min_allocator.h"

template <class Allocator>
void
test()
{
    typedef std::vector<bool, Allocator> C;

    static_assert((std::is_same<typename C::value_type, bool>::value), "");
    static_assert((std::is_same<typename C::value_type, typename Allocator::value_type>::value), "");
    static_assert((std::is_same<typename C::allocator_type, Allocator>::value), "");
    static_assert((std::is_same<typename C::size_type, typename std::allocator_traits<Allocator>::size_type>::value), "");
    static_assert((std::is_same<typename C::difference_type, typename std::allocator_traits<Allocator>::difference_type>::value), "");

    static_assert((std::is_signed<typename C::difference_type>::value), "");
    static_assert((std::is_unsigned<typename C::size_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
        typename std::iterator_traits<typename C::iterator>::difference_type>::value), "");
    static_assert((std::is_same<typename C::difference_type,
        typename std::iterator_traits<typename C::const_iterator>::difference_type>::value), "");

   static_assert((std::is_same<
        typename std::iterator_traits<typename C::iterator>::iterator_category,
        std::random_access_iterator_tag>::value), "");
    static_assert((std::is_same<
        typename std::iterator_traits<typename C::const_iterator>::iterator_category,
        std::random_access_iterator_tag>::value), "");
    static_assert((std::is_same<
        typename C::reverse_iterator,
        std::reverse_iterator<typename C::iterator> >::value), "");
    static_assert((std::is_same<
        typename C::const_reverse_iterator,
        std::reverse_iterator<typename C::const_iterator> >::value), "");
#if !defined(_LIBCPP_VERSION) || defined(_LIBCPP_ABI_BITSET_VECTOR_BOOL_CONST_SUBSCRIPT_RETURN_BOOL)
    static_assert(std::is_same<typename C::const_reference, bool>::value, "");
#endif
}

int main(int, char**)
{
    test<test_allocator<bool> >();
    test<std::allocator<bool> >();
    static_assert((std::is_same<std::vector<bool>::allocator_type,
                                std::allocator<bool> >::value), "");
#if TEST_STD_VER >= 11
    test<min_allocator<bool> >();
#endif

  return 0;
}
