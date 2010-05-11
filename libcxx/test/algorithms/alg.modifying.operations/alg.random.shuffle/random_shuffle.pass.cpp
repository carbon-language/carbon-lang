//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter> 
//   requires ShuffleIterator<Iter> 
//   void
//   random_shuffle(Iter first, Iter last);

#include <algorithm>

#include "../../iterators.h"

int main()
{
    int ia[] = {1, 2, 3, 4};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    std::random_shuffle(ia, ia+sa);
}
