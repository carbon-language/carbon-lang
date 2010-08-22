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

// size_type max_size() const;

#include <scoped_allocator>
#include <cassert>

#include "../allocators.h"

int main()
{
#ifdef _LIBCPP_MOVE

    {
        typedef std::scoped_allocator_adaptor<A1<int>> A;
        const A a(A1<int>(100));
        assert(a.max_size() == 100);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>> A;
        const A a(A1<int>(20), A2<int>());
        assert(a.max_size() == 20);
    }
    {
        typedef std::scoped_allocator_adaptor<A1<int>, A2<int>, A3<int>> A;
        const A a(A1<int>(200), A2<int>(), A3<int>());
        assert(a.max_size() == 200);
    }

#endif  // _LIBCPP_MOVE
}
