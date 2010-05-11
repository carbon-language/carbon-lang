//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template<class E> class initializer_list;

// initializer_list();

#include <initializer_list>
#include <cassert>

struct A {};

int main()
{
    std::initializer_list<A> il;
    assert(il.size() == 0);
}
