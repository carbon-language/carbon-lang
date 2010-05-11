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

// inner_allocator_type& inner_allocator();
// const inner_allocator_type& inner_allocator() const;

#include <memory>
#include <cassert>

#include "../allocators.h"

int main()
{
#ifdef _LIBCPP_MOVE

    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        A a(A1<int>(5));
        assert(a.inner_allocator() == a);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        A a(A1<int>(5), A2<int>(6));
        assert(a.inner_allocator() == std::scoped_allocator_adaptor<A2<int>>(A2<int>(6)));
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        A a(A1<int>(5), A2<int>(6), A3<int>(8));
        assert((a.inner_allocator() ==
            std::scoped_allocator_adaptor<A2<int>, A3<int>>(A2<int>(6), A3<int>(8))));
    }

#endif
}
