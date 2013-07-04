//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// http://llvm.org/bugs/show_bug.cgi?id=16538

#include <unordered_map>

struct Key {
  template <typename T> Key(const T&) {}
  bool operator== (const Key&) const { return true; }
};

namespace std
{
    template <>
    struct hash<Key>
    {
        size_t operator()(Key const &) const {return 0;}
    };
}

int
main()
{
    std::unordered_map<Key, int>::iterator it =
        std::unordered_map<Key, int>().find(Key(0));
}
