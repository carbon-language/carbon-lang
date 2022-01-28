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

// scoped_allocator_adaptor& operator=(const scoped_allocator_adaptor& other);


#include <scoped_allocator>
#include <cassert>

#include "test_macros.h"
#include "allocators.h"

int main(int, char**)
{
    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a1(A1<int>(3));
        A aN;
        A1<int>::copy_called = false;
        A1<int>::move_called = false;
        aN = a1;
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(aN == a1);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a1(A1<int>(4), A2<int>(5));
        A aN;
        A1<int>::copy_called = false;
        A1<int>::move_called = false;
        A2<int>::copy_called = false;
        A2<int>::move_called = false;
        aN = a1;
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(aN == a1);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a1(A1<int>(4), A2<int>(5), A3<int>(6));
        A aN;
        A1<int>::copy_called = false;
        A1<int>::move_called = false;
        A2<int>::copy_called = false;
        A2<int>::move_called = false;
        A3<int>::copy_called = false;
        A3<int>::move_called = false;
        aN = a1;
        assert(A1<int>::copy_called == true);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == true);
        assert(A2<int>::move_called == false);
        assert(A3<int>::copy_called == true);
        assert(A3<int>::move_called == false);
        assert(aN == a1);
    }

  return 0;
}
