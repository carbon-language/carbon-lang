//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class OuterAlloc, class... InnerAllocs>
//   class scoped_allocator_adaptor

// scoped_allocator_adaptor();

#include <scoped_allocator>
#include <cassert>

#include "../allocators.h"

int main()
{
#ifdef _LIBCPP_MOVE

    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a;
        assert(a.outer_allocator() == A1<int>());
        assert(a.inner_allocator() == a);
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == false);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a;
        assert(a.outer_allocator() == A1<int>());
        assert(a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>>());
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == false);
        assert(A2<int>::move_called == false);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a;
        assert(a.outer_allocator() == A1<int>());
        assert((a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>, A3<int>>()));
        assert(A1<int>::copy_called == false);
        assert(A1<int>::move_called == false);
        assert(A2<int>::copy_called == false);
        assert(A2<int>::move_called == false);
        assert(A3<int>::copy_called == false);
        assert(A3<int>::move_called == false);
    }

#endif
}
