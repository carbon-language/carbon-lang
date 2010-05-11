//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// back_insert_iterator

// explicit back_insert_iterator(Cont& x);

#include <iterator>
#include <vector>

template <class C>
void
test(C c)
{
    std::back_insert_iterator<C> i(c);
}

int main()
{
    test(std::vector<int>());
}
