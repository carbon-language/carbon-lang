//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// front_insert_iterator

// explicit front_insert_iterator(Cont& x);

// test for explicit

#include <iterator>
#include <list>

int main()
{
    std::front_insert_iterator<std::list<int> > i = std::list<int>();
}
