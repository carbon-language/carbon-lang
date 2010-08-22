//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <set>

// class multiset

// void insert(initializer_list<value_type> il);

#include <set>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    typedef std::multiset<int> C;
    typedef C::value_type V;
    C m = {10, 8};
    m.insert({1, 2, 3, 4, 5, 6});
    assert(m.size() == 8);
    assert(distance(m.begin(), m.end()) == m.size());
    C::const_iterator i = m.cbegin();
    assert(*i == V(1));
    assert(*++i == V(2));
    assert(*++i == V(3));
    assert(*++i == V(4));
    assert(*++i == V(5));
    assert(*++i == V(6));
    assert(*++i == V(8));
    assert(*++i == V(10));
#endif  // _LIBCPP_MOVE
}
