//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <BackInsertionContainer Cont> 
//   front_insert_iterator<Cont>
//   front_inserter(Cont& x);

#include <iterator>
#include <list>
#include <cassert>

template <class C>
void
test(C c)
{
    std::front_insert_iterator<C> i = std::front_inserter(c);
    i = 0;
    assert(c.size() == 1);
    assert(c.front() == 0);
}

int main()
{
    test(std::list<int>());
}
