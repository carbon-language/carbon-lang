//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <scoped_allocator>

// template <class OtherAlloc, class ...InnerAlloc>
//   class scoped_allocator_adaptor

// template <class U1, class U2>
// void scoped_allocator_adaptor::construct(pair<U1, U2>*, pair<T1, T2>const&)

#include <scoped_allocator>
#include <type_traits>
#include <utility>
#include <tuple>
#include <cassert>
#include <cstdlib>
#include "uses_alloc_types.h"
#include "controlled_allocators.h"

#include "test_macros.h"


void test_no_inner_alloc()
{
    using VoidAlloc = CountingAllocator<void>;
    AllocController P;
    {
        using T = UsesAllocatorV1<VoidAlloc, 1>;
        using U = UsesAllocatorV2<VoidAlloc, 1>;
        using Pair = std::pair<T, U>;
        using PairIn = std::pair<int&, int const&&>;
        int x = 42;
        const int y = 101;
        using Alloc = CountingAllocator<Pair>;
        using SA = std::scoped_allocator_adaptor<Alloc>;
        static_assert(std::uses_allocator<T, CountingAllocator<T> >::value, "");
        Pair * ptr = (Pair*)std::malloc(sizeof(Pair));
        assert(ptr != nullptr);
        Alloc CA(P);
        SA A(CA);
        const PairIn in(x, std::move(y));
        A.construct(ptr, in);
        assert(checkConstruct<int&>(ptr->first, UA_AllocArg, CA));
        assert(checkConstruct<int const&>(ptr->second, UA_AllocLast, CA));
        assert((P.checkConstruct<std::piecewise_construct_t const&,
                                 std::tuple<std::allocator_arg_t, SA&, int&>&&,
                                 std::tuple<int const&, SA&>&&
              >(CA, ptr)));
        A.destroy(ptr);
        std::free(ptr);

    }
    P.reset();
    {
        using T = UsesAllocatorV3<VoidAlloc, 1>;
        using U = NotUsesAllocator<VoidAlloc, 1>;
        using Pair = std::pair<T, U>;
        using PairIn = std::pair<int, int const&>;
        int x = 42;
        const int y = 101;
        using Alloc = CountingAllocator<Pair>;
        using SA = std::scoped_allocator_adaptor<Alloc>;
        static_assert(std::uses_allocator<T, CountingAllocator<T> >::value, "");
        Pair * ptr = (Pair*)std::malloc(sizeof(Pair));
        assert(ptr != nullptr);
        Alloc CA(P);
        SA A(CA);
        const PairIn in(x, y);
        A.construct(ptr, in);
        assert(checkConstruct<int const&>(ptr->first, UA_AllocArg, CA));
        assert(checkConstruct<int const&>(ptr->second, UA_None));
        assert((P.checkConstruct<std::piecewise_construct_t const&,
                                 std::tuple<std::allocator_arg_t, SA&, int const&>&&,
                                 std::tuple<int const&>&&
                   >(CA, ptr)));
        A.destroy(ptr);
        std::free(ptr);
    }
}

void test_with_inner_alloc()
{
    using VoidAlloc2 = CountingAllocator<void, 2>;

    AllocController POuter;
    AllocController PInner;
    {
        using T = UsesAllocatorV1<VoidAlloc2, 1>;
        using U = UsesAllocatorV2<VoidAlloc2, 1>;
        using Pair = std::pair<T, U>;
        using PairIn = std::pair<int&, int const&&>;
        int x = 42;
        int y = 101;
        using Outer = CountingAllocator<Pair, 1>;
        using Inner = CountingAllocator<Pair, 2>;
        using SA = std::scoped_allocator_adaptor<Outer, Inner>;
        using SAInner = std::scoped_allocator_adaptor<Inner>;
        static_assert(!std::uses_allocator<T, Outer>::value, "");
        static_assert(std::uses_allocator<T, Inner>::value, "");
        Pair * ptr = (Pair*)std::malloc(sizeof(Pair));
        assert(ptr != nullptr);
        Outer O(POuter);
        Inner I(PInner);
        SA A(O, I);
        const PairIn in(x, std::move(y));
        A.construct(ptr, in);
        assert(checkConstruct<int&>(ptr->first, UA_AllocArg, I));
        assert(checkConstruct<int const&>(ptr->second, UA_AllocLast));
        assert((POuter.checkConstruct<std::piecewise_construct_t const&,
                                 std::tuple<std::allocator_arg_t, SAInner&, int&>&&,
                                 std::tuple<int const&, SAInner&>&&
              >(O, ptr)));
        A.destroy(ptr);
        std::free(ptr);
    }
    PInner.reset();
    POuter.reset();
    {
        using T = UsesAllocatorV3<VoidAlloc2, 1>;
        using U = NotUsesAllocator<VoidAlloc2, 1>;
        using Pair = std::pair<T, U>;
        using PairIn = std::pair<int, int const &>;
        int x = 42;
        int y = 101;
        using Outer = CountingAllocator<Pair, 1>;
        using Inner = CountingAllocator<Pair, 2>;
        using SA = std::scoped_allocator_adaptor<Outer, Inner>;
        using SAInner = std::scoped_allocator_adaptor<Inner>;
        static_assert(!std::uses_allocator<T, Outer>::value, "");
        static_assert(std::uses_allocator<T, Inner>::value, "");
        Pair * ptr = (Pair*)std::malloc(sizeof(Pair));
        assert(ptr != nullptr);
        Outer O(POuter);
        Inner I(PInner);
        SA A(O, I);
        const PairIn in(x, y);
        A.construct(ptr, in);
        assert(checkConstruct<int const&>(ptr->first, UA_AllocArg, I));
        assert(checkConstruct<int const&>(ptr->second, UA_None));
        assert((POuter.checkConstruct<std::piecewise_construct_t const&,
                                 std::tuple<std::allocator_arg_t, SAInner&, int const&>&&,
                                 std::tuple<int const&>&&
              >(O, ptr)));
        A.destroy(ptr);
        std::free(ptr);
    }
}
int main(int, char**) {
    test_no_inner_alloc();
    test_with_inner_alloc();

  return 0;
}
