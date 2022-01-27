//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <unordered_map>

// template <class Key, class T, class Hash, class Pred, class Allocator, class Predicate>
//   typename unordered_map<Key, T, Hash, Pred, Allocator>::size_type
//   erase_if(unordered_map<Key, T, Hash, Pred, Allocator>& c, Predicate pred);

#include <unordered_map>
#include <algorithm>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

using Init = std::initializer_list<int>;
template <typename M>
M make (Init vals)
{
    M ret;
    for (int v : vals)
        ret[static_cast<typename M::key_type>(v)] = static_cast<typename M::mapped_type>(v + 10);
    return ret;
}

template <typename M, typename Pred>
void test0(Init vals, Pred p, Init expected, size_t expected_erased_count) {
  M s = make<M>(vals);
  ASSERT_SAME_TYPE(typename M::size_type, decltype(std::erase_if(s, p)));
  assert(expected_erased_count == std::erase_if(s, p));
  M e = make<M>(expected);
  assert((std::is_permutation(s.begin(), s.end(), e.begin(), e.end())));
}

template <typename S>
void test()
{
    auto is1 = [](auto v) { return v.first == 1;};
    auto is2 = [](auto v) { return v.first == 2;};
    auto is3 = [](auto v) { return v.first == 3;};
    auto is4 = [](auto v) { return v.first == 4;};
    auto True  = [](auto) { return true; };
    auto False = [](auto) { return false; };

    test0<S>({}, is1, {}, 0);

    test0<S>({1}, is1, {}, 1);
    test0<S>({1}, is2, {1}, 0);

    test0<S>({1, 2}, is1, {2}, 1);
    test0<S>({1, 2}, is2, {1}, 1);
    test0<S>({1, 2}, is3, {1, 2}, 0);

    test0<S>({1, 2, 3}, is1, {2, 3}, 1);
    test0<S>({1, 2, 3}, is2, {1, 3}, 1);
    test0<S>({1, 2, 3}, is3, {1, 2}, 1);
    test0<S>({1, 2, 3}, is4, {1, 2, 3}, 0);

    test0<S>({1, 2, 3}, True, {}, 3);
    test0<S>({1, 2, 3}, False, {1, 2, 3}, 0);
}

int main(int, char**)
{
    test<std::unordered_map<int, int>>();
    test<std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, min_allocator<std::pair<const int, int>>>> ();
    test<std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, test_allocator<std::pair<const int, int>>>> ();

    test<std::unordered_map<long, short>>();
    test<std::unordered_map<short, double>>();

  return 0;
}
