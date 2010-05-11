//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// void insert(initializer_list<value_type> il);

#include <map>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    typedef std::pair<const int, double> V;
    std::map<int, double> m =
                            {
                                {1, 1},
                                {1, 1.5},
                                {1, 2},
                                {3, 1},
                                {3, 1.5},
                                {3, 2}
                            };
    m.insert({
                 {2, 1},
                 {2, 1.5},
                 {2, 2},
             });
    assert(m.size() == 3);
    assert(distance(m.begin(), m.end()) == 3);
    assert(*m.begin() == V(1, 1));
    assert(*next(m.begin()) == V(2, 1));
    assert(*next(m.begin(), 2) == V(3, 1));
#endif
}
