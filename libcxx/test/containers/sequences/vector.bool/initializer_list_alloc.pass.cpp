//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// vector(initializer_list<value_type> il, const Allocator& a = allocator_type());

#include <vector>
#include <cassert>

#include "../../test_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::vector<int, test_allocator<int>> d({true, false, false, true}, test_allocator<int>(3));
    assert(d.get_allocator() == test_allocator<int>(3));
    assert(d.size() == 4);
    assert(d[0] == true);
    assert(d[1] == false);
    assert(d[2] == false);
    assert(d[3] == true);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
