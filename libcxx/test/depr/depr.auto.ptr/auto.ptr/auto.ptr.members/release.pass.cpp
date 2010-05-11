//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class X> class auto_ptr;

// X* release() throw();

#include <memory>
#include <cassert>

#include "../A.h"

void
test()
{
    {
    A* p = new A(1);
    std::auto_ptr<A> ap(p);
    A* p2 = ap.release();
    assert(p2 == p);
    assert(ap.get() == 0);
    delete p;
    }
    assert(A::count == 0);
}

int main()
{
    test();
}
