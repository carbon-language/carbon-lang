//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// template <class Key, class T, class Compare = less<Key>,
//           class Allocator = allocator<pair<const Key, T>>>
// class map

// http://llvm.org/bugs/show_bug.cgi?id=16538

#include <map>

struct Key {
  template <typename T> Key(const T&) {}
  bool operator< (const Key&) const { return false; }
};

int
main()
{
    std::map<Key, int>::iterator it = std::map<Key, int>().find(Key(0));
}
