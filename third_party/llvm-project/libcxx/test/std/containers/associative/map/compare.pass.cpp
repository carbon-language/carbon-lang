//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// template <class Key, class T, class Compare = less<Key>,
//           class Allocator = allocator<pair<const Key, T>>>
// class map

// https://llvm.org/PR16538
// https://llvm.org/PR16549

#include <map>
#include <utility>
#include <cassert>

#include "test_macros.h"

struct Key {
  template <typename T> Key(const T&) {}
  bool operator< (const Key&) const { return false; }
};

int main(int, char**)
{
    typedef std::map<Key, int> MapT;
    typedef MapT::iterator Iter;
    typedef std::pair<Iter, bool> IterBool;
    {
        MapT m_empty;
        MapT m_contains;
        m_contains[Key(0)] = 42;

        Iter it = m_empty.find(Key(0));
        assert(it == m_empty.end());
        it = m_contains.find(Key(0));
        assert(it != m_contains.end());
    }
    {
        MapT map;
        IterBool result = map.insert(std::make_pair(Key(0), 42));
        assert(result.second);
        assert(result.first->second == 42);
        IterBool result2 = map.insert(std::make_pair(Key(0), 43));
        assert(!result2.second);
        assert(map[Key(0)] == 42);
    }

  return 0;
}
