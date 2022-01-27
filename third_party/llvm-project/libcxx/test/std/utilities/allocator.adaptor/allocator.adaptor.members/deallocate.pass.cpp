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
//   class scoped_allocator_adaptor

// void deallocate(pointer p, size_type n);

#include <scoped_allocator>
#include <cassert>

#include "test_macros.h"
#include "allocators.h"

int main(int, char**)
{

    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a;
        a.deallocate((int*)10, 20);
        assert((A1<int>::deallocate_called == std::pair<int*, std::size_t>((int*)10, 20)));
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a;
        a.deallocate((int*)10, 20);
        assert((A1<int>::deallocate_called == std::pair<int*, std::size_t>((int*)10, 20)));
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a;
        a.deallocate((int*)10, 20);
        assert((A1<int>::deallocate_called == std::pair<int*, std::size_t>((int*)10, 20)));
    }


  return 0;
}
