//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class OuterAlloc, class... InnerAllocs>
//   class scoped_allocator_adaptor

// template <class T> void destroy(T* p);

#include <memory>
#include <cassert>
#include <string>

#include "../allocators.h"

struct B
{
    static bool constructed;

    B() {constructed = true;}
    ~B() {constructed = false;}
};

bool B::constructed = false;

int main()
{
#ifdef _LIBCPP_MOVE

    {
        typedef std::scoped_allocator_adaptor<A1<B>> A;
        A a;
        char buf[100];
        typedef B S;
        S* s = (S*)buf;
        assert(!S::constructed);
        a.construct(s);
        assert(S::constructed);
        a.destroy(s);
        assert(!S::constructed);
    }

    {
        typedef std::scoped_allocator_adaptor<A3<B>, A1<B>> A;
        A a;
        char buf[100];
        typedef B S;
        S* s = (S*)buf;
        assert(!S::constructed);
        assert(!A3<S>::constructed);
        assert(!A3<S>::destroy_called);
        a.construct(s);
        assert(S::constructed);
        assert(A3<S>::constructed);
        assert(!A3<S>::destroy_called);
        a.destroy(s);
        assert(!S::constructed);
        assert(A3<S>::constructed);
        assert(A3<S>::destroy_called);
    }

#endif
}
