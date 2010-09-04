//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// vector(initializer_list<value_type> il);

#include <vector>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::vector<int> d = {3, 4, 5, 6};
    assert(d.size() == 4);
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
