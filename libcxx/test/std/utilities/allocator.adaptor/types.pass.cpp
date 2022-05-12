//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <memory>

// template <class OuterAlloc, class... InnerAllocs>
// class scoped_allocator_adaptor
//     : public OuterAlloc
// {
// public:
//     typedef OuterAlloc outer_allocator_type;
//     typedef typename OuterTraits::size_type size_type;
//     typedef typename OuterTraits::difference_type difference_type;
//     typedef typename OuterTraits::pointer pointer;
//     typedef typename OuterTraits::const_pointer const_pointer;
//     typedef typename OuterTraits::void_pointer void_pointer;
//     typedef typename OuterTraits::const_void_pointer const_void_pointer;
// };

#include <scoped_allocator>
#include <type_traits>

#include "test_macros.h"
#include "allocators.h"

int main(int, char**)
{
    static_assert((std::is_base_of<
        A1<int>,
        std::scoped_allocator_adaptor<A1<int>>
        >::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::outer_allocator_type,
        A1<int>>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::size_type,
        std::make_unsigned<std::ptrdiff_t>::type>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::difference_type,
        std::ptrdiff_t>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::pointer,
        int*>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::const_pointer,
        const int*>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::void_pointer,
        void*>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A1<int>>::const_void_pointer,
        const void*>::value), "");

    static_assert((std::is_base_of<
        A2<int>,
        std::scoped_allocator_adaptor<A2<int>, A1<int>>
        >::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A2<int>, A1<int>>::outer_allocator_type,
        A2<int>>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A2<int>, A1<int>>::size_type,
        unsigned>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A2<int>, A1<int>>::difference_type,
        int>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A2<int>, A1<int>>::pointer,
        int*>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A2<int>, A1<int>>::const_pointer,
        const int*>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A2<int>, A1<int>>::void_pointer,
        void*>::value), "");

    static_assert((std::is_same<
        std::scoped_allocator_adaptor<A2<int>, A1<int>>::const_void_pointer,
        const void*>::value), "");

  return 0;
}
