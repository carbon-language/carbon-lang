//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// reverse_iterator();

#include <iterator>

#include "../../../../iterators.h"

template <class It>
void
test()
{
    std::reverse_iterator<It> r;
}

int main()
{
    test<bidirectional_iterator<const char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();
    test<const char*>();
}
