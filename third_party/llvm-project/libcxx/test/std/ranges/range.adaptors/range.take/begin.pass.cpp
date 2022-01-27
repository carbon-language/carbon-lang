//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr auto begin() requires (!simple-view<V>);
// constexpr auto begin() const requires range<const V>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"
#include "types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // sized_range && random_access_iterator
  {
    std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView(buffer), 4);
    assert(tv.begin() == SizedRandomAccessView(buffer).begin());
    ASSERT_SAME_TYPE(decltype(tv.begin()), RandomAccessIter);
  }

  {
    const std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView(buffer), 4);
    assert(tv.begin() == SizedRandomAccessView(buffer).begin());
    ASSERT_SAME_TYPE(decltype(tv.begin()), RandomAccessIter);
  }

  // sized_range && !random_access_iterator
  {
    std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<ForwardIter>(ForwardIter(buffer), 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<ForwardIter>);
  }

  {
    const std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<ForwardIter>(ForwardIter(buffer), 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<ForwardIter>);
  }

  // !sized_range
  {
    std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<int*>(buffer, 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<int*>);
  }

  {
    const std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
    assert(tv.begin() == std::counted_iterator<int*>(buffer, 4));
    ASSERT_SAME_TYPE(decltype(tv.begin()), std::counted_iterator<int*>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
