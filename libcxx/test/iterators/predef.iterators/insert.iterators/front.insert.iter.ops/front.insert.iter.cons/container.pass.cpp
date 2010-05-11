//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// front_insert_iterator

// explicit front_insert_iterator(Cont& x);

#include <iterator>
#include <list>

template <class C>
void
test(C c)
{
    std::front_insert_iterator<C> i(c);
}

int main()
{
    test(std::list<int>());
}
