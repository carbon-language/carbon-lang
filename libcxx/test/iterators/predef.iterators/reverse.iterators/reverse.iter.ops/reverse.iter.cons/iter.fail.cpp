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

// explicit reverse_iterator(Iter x);

// test explicit

#include <iterator>

template <class It>
void
test(It i)
{
    std::reverse_iterator<It> r = i;
}

int main()
{
    const char s[] = "123";
    test(s);
}
