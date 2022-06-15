//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr sentinel_t<V> base() const;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>
#include "test_iterators.h"
#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterSentinel = std::ranges::sentinel_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return FilterView(std::move(view), pred);
  };

  std::array<int, 5> array{0, 1, 2, 3, 4};
  FilterView view = make_filter_view(array.begin(), array.end(), AlwaysTrue{});

  FilterSentinel const sent = view.end();
  std::same_as<Sentinel> decltype(auto) result = sent.base();
  assert(base(base(result)) == array.end());
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
