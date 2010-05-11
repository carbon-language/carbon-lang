//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter, Callable<auto, Iter::difference_type> Rand> 
//   requires ShuffleIterator<Iter> 
//         && Convertible<Rand::result_type, Iter::difference_type> 
//   void
//   random_shuffle(Iter first, Iter last, Rand&& rand);

#include <algorithm>

#include "../../iterators.h"

struct gen
{
    int operator()(int n)
    {
        return 0;
    }
};

int main()
{
    int ia[] = {1, 2, 3, 4};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);
    gen r;
    std::random_shuffle(ia, ia+sa, r);
}
