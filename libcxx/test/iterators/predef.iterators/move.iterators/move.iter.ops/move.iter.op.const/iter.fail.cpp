//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// explicit move_iterator(Iter );

// test explicit

#include <iterator>

template <class It>
void
test(It i)
{
    std::move_iterator<It> r = i;
}

int main()
{
    char s[] = "123";
    test(s);
}
