//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// template <class C> auto begin(const C& c) -> decltype(c.begin());

#include <vector>
#include <cassert>

int main()
{
    int ia[] = {1, 2, 3};
    const std::vector<int> v(ia, ia + sizeof(ia)/sizeof(ia[0]));
    std::vector<int>::const_iterator i = begin(v);
    assert(*i == 1);
}
