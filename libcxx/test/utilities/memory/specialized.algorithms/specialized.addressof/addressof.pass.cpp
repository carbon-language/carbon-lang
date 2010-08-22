//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <ObjectType T> T* addressof(T& r);

#include <memory>
#include <cassert>

struct A
{
    void operator&() const {}
};

int main()
{
    int i;
    double d;
    assert(std::addressof(i) == &i);
    assert(std::addressof(d) == &d);
    A* tp = new A;
    const A* ctp = tp;
    assert(std::addressof(*tp) == tp);
    assert(std::addressof(*ctp) == tp);
    delete tp;
}
