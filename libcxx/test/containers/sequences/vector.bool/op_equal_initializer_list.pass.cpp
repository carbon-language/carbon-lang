//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// vector& operator=(initializer_list<value_type> il);

#include <vector>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::vector<bool> d;
    d = {true, false, false, true};
    assert(d.size() == 4);
    assert(d[0] == true);
    assert(d[1] == false);
    assert(d[2] == false);
    assert(d[3] == true);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
