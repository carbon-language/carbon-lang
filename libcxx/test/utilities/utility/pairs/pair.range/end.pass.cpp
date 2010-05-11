//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template <class InputIterator>
//     InputIterator
//     end(const std::pair<InputIterator, InputIterator>& p);

#include <utility>
#include <iterator>
#include <cassert>

int main()
{
    {
        typedef std::pair<int*, int*> P;
        int a[3] = {0};
        P p(std::begin(a), std::end(a));
        assert(std::end(p) == a+3);
    }
}
