//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class T> 
//   T
//   max(initializer_list<T> t);

#include <algorithm>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    int i = std::max({2, 3, 1});
    assert(i == 3);
    i = std::max({2, 1, 3});
    assert(i == 3);
    i = std::max({3, 1, 2});
    assert(i == 3);
    i = std::max({3, 2, 1});
    assert(i == 3);
    i = std::max({1, 2, 3});
    assert(i == 3);
    i = std::max({1, 3, 2});
    assert(i == 3);
#endif
}
