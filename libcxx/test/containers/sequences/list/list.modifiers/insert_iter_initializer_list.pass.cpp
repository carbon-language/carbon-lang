//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// iterator insert(const_iterator p, initializer_list<value_type> il);

#include <list>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    std::list<int> d(10, 1);
    std::list<int>::iterator i = d.insert(next(d.cbegin(), 2), {3, 4, 5, 6});
    assert(d.size() == 14);
    assert(i == next(d.begin(), 2));
    i = d.begin();
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 3);
    assert(*i++ == 4);
    assert(*i++ == 5);
    assert(*i++ == 6);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
    assert(*i++ == 1);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
