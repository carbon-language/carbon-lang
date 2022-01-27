//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::advance(it, n, sent)

#include <iterator>

#include <cassert>
#include <climits>
#include <concepts>
#include <cstddef>

#include "test_iterators.h"
#include "../types.h"

template <bool Count, typename It>
constexpr void check_forward(int* first, int* last, std::iter_difference_t<It> n, int* expected) {
  using Difference = std::iter_difference_t<It>;
  Difference const M = (expected - first); // expected travel distance

  {
    It it(first);
    auto sent = sentinel_wrapper(It(last));
    std::same_as<Difference> auto result = std::ranges::advance(it, n, sent);
    assert(result == n - M);
    assert(base(it) == expected);
  }

  // Count operations
  if constexpr (Count) {
    auto it = stride_counting_iterator(It(first));
    auto sent = sentinel_wrapper(stride_counting_iterator(It(last)));
    (void)std::ranges::advance(it, n, sent);
    // We don't have a sized sentinel, so we have to increment one-by-one
    // regardless of the iterator category.
    assert(it.stride_count() == M);
    assert(it.stride_displacement() == M);
  }
}

template <typename It>
constexpr void check_forward_sized_sentinel(int* first, int* last, std::iter_difference_t<It> n, int* expected) {
  using Difference = std::iter_difference_t<It>;
  Difference const size = (last - first);
  Difference const M = (expected - first); // expected travel distance

  {
    It it(first);
    auto sent = distance_apriori_sentinel(size);
    std::same_as<Difference> auto result = std::ranges::advance(it, n, sent);
    assert(result == n - M);
    assert(base(it) == expected);
  }

  // Count operations
  {
    auto it = stride_counting_iterator(It(first));
    auto sent = distance_apriori_sentinel(size);
    (void)std::ranges::advance(it, n, sent);
    if constexpr (std::random_access_iterator<It>) {
      assert(it.stride_count() <= 1);
      assert(it.stride_displacement() <= 1);
    } else {
      assert(it.stride_count() == M);
      assert(it.stride_displacement() == M);
    }
  }
}

template <typename It>
constexpr void check_backward(int* first, int* last, std::iter_difference_t<It> n, int* expected) {
  static_assert(std::random_access_iterator<It>, "This test doesn't support non random access iterators");
  using Difference = std::iter_difference_t<It>;
  Difference const M = (expected - last); // expected travel distance (which is negative)

  {
    It it(last);
    It sent(first);
    std::same_as<Difference> auto result = std::ranges::advance(it, n, sent);
    assert(result == n - M);
    assert(base(it) == expected);
  }

  // Count operations
  {
    auto it = stride_counting_iterator(It(last));
    auto sent = stride_counting_iterator(It(first));
    (void)std::ranges::advance(it, n, sent);
    assert(it.stride_count() <= 1);
    assert(it.stride_displacement() <= 1);
  }
}

struct iota_iterator {
    using difference_type = int;
    using value_type = int;

    constexpr int operator*() const { return x; }
    constexpr iota_iterator& operator++() { ++x; return *this; }
    constexpr iota_iterator operator++(int) { ++x; return iota_iterator{x - 1}; }
    constexpr bool operator==(const iota_iterator&) const = default;
    constexpr int operator-(const iota_iterator& that) const { return x - that.x; }
    constexpr iota_iterator& operator--() { --x; return *this; }
    constexpr iota_iterator operator--(int) { --x; return iota_iterator{x + 1}; }

    int x;
};
static_assert(std::bidirectional_iterator<iota_iterator>);
static_assert(std::sized_sentinel_for<iota_iterator, iota_iterator>);

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (int size = 0; size != 10; ++size) {
    for (int n = 0; n != 20; ++n) {

      {
        int* expected = n > size ? range + size : range + n;
        check_forward<false, cpp17_input_iterator<int*>>(  range, range+size, n, expected);
        check_forward<false, cpp20_input_iterator<int*>>(  range, range+size, n, expected);
        check_forward<true,  forward_iterator<int*>>(      range, range+size, n, expected);
        check_forward<true,  bidirectional_iterator<int*>>(range, range+size, n, expected);
        check_forward<true,  random_access_iterator<int*>>(range, range+size, n, expected);
        check_forward<true,  contiguous_iterator<int*>>(   range, range+size, n, expected);
        check_forward<true,  int*>(                        range, range+size, n, expected);

        check_forward_sized_sentinel<cpp17_input_iterator<int*>>(  range, range+size, n, expected);
        check_forward_sized_sentinel<cpp20_input_iterator<int*>>(  range, range+size, n, expected);
        check_forward_sized_sentinel<forward_iterator<int*>>(      range, range+size, n, expected);
        check_forward_sized_sentinel<bidirectional_iterator<int*>>(range, range+size, n, expected);
        check_forward_sized_sentinel<random_access_iterator<int*>>(range, range+size, n, expected);
        check_forward_sized_sentinel<contiguous_iterator<int*>>(   range, range+size, n, expected);
        check_forward_sized_sentinel<int*>(                        range, range+size, n, expected);
      }

      {
        // Note that we can only test ranges::advance with a negative n for iterators that
        // are sized sentinels for themselves, because ranges::advance is UB otherwise.
        // In particular, that excludes bidirectional_iterators since those are not sized sentinels.
        // TODO: Enable these tests once we fix the bug in ranges::advance
        // int* expected = n > size ? range : range + size - n;
        // check_backward<random_access_iterator<int*>>(range, range+size, -n, expected);
        // check_backward<contiguous_iterator<int*>>(   range, range+size, -n, expected);
        // check_backward<int*>(                        range, range+size, -n, expected);
      }
    }
  }

  // regression-test that INT_MIN doesn't cause any undefined behavior
  {
    auto i = iota_iterator{+1};
    assert(std::ranges::advance(i, INT_MIN, iota_iterator{-2}) == INT_MIN+3);
    assert(i == iota_iterator{-2});
    i = iota_iterator{+1};
    assert(std::ranges::advance(i, -2, iota_iterator{INT_MIN+1}) == 0);
    assert(i == iota_iterator{-1});
    i = iota_iterator{+1};
    assert(std::ranges::advance(i, INT_MIN, iota_iterator{INT_MIN+1}) == 0);
    assert(i == iota_iterator{INT_MIN+1});
  }

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
