//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// list(initializer_list<value_type> il);

#include <list>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::list<int> d = {3, 4, 5, 6};
    assert(d.size() == 4);
    std::list<int>::iterator i = d.begin();
    assert(*i++ == 3);
    assert(*i++ == 4);
    assert(*i++ == 5);
    assert(*i++ == 6);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
