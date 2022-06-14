//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <unordered_set>

// class unordered_set

// insert_return_type insert(node_type&&);

#include <unordered_set>
#include <type_traits>
#include "test_macros.h"
#include "min_allocator.h"

template <class Container>
typename Container::node_type
node_factory(typename Container::key_type const& key)
{
    static Container c;
    auto p = c.insert(key);
    assert(p.second);
    return c.extract(p.first);
}

template <class Container>
void test(Container& c)
{
    auto* nf = &node_factory<Container>;

    for (int i = 0; i != 10; ++i)
    {
        typename Container::node_type node = nf(i);
        assert(!node.empty());
        typename Container::insert_return_type irt = c.insert(std::move(node));
        assert(node.empty());
        assert(irt.inserted);
        assert(irt.node.empty());
        assert(*irt.position == i);
    }

    assert(c.size() == 10);

    { // Insert empty node.
        typename Container::node_type def;
        auto irt = c.insert(std::move(def));
        assert(def.empty());
        assert(!irt.inserted);
        assert(irt.node.empty());
        assert(irt.position == c.end());
    }

    { // Insert duplicate node.
        typename Container::node_type dupl = nf(0);
        auto irt = c.insert(std::move(dupl));
        assert(dupl.empty());
        assert(!irt.inserted);
        assert(!irt.node.empty());
        assert(irt.position == c.find(0));
        assert(irt.node.value() == 0);
    }

    assert(c.size() == 10);

    for (int i = 0; i != 10; ++i)
    {
        assert(c.count(i) == 1);
    }
}

int main(int, char**)
{
    std::unordered_set<int> m;
    test(m);
    std::unordered_set<int, std::hash<int>, std::equal_to<int>, min_allocator<int>> m2;
    test(m2);

  return 0;
}
