//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// void insert(initializer_list<value_type> il);

#include <unordered_map>
#include <string>
#include <cassert>

#include "../../../iterators.h"

int main()
{
#ifdef _LIBCPP_MOVE
    {
        typedef std::unordered_map<int, std::string> C;
        typedef std::pair<int, std::string> P;
        C c;
        c.insert(
                    {
                        P(1, "one"),
                        P(2, "two"),
                        P(3, "three"),
                        P(4, "four"),
                        P(1, "four"),
                        P(2, "four"),
                    }
                );
        assert(c.size() == 4);
        assert(c.at(1) == "one");
        assert(c.at(2) == "two");
        assert(c.at(3) == "three");
        assert(c.at(4) == "four");
    }
#endif
}
