//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts, libcpp-has-no-incomplete-ranges

// <memory>
//
// template<input_iterator I, sentinel-for<I> S1, nothrow-forward-iterator O, nothrow-sentinel-for<O> S2>
//   requires constructible_from<iter_value_t<O>, iter_reference_t<I>>
// uninitialized_copy_result<I, O> ranges::uninitialized_copy(I ifirst, S1 ilast, O ofirst, S2 olast); // since C++20
//
// template<input_range IR, nothrow-forward-range OR>
//   requires constructible_from<range_value_t<OR>, range_reference_t<IR>>
// uninitialized_copy_result<borrowed_iterator_t<IR>, borrowed_iterator_t<OR>> ranges::uninitialized_copy(IR&& in_range, OR&& out_range); // since C++20

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>

#include "../buffer.h"
#include "../counted.h"
#include "test_macros.h"
#include "test_iterators.h"

// TODO(varconst): consolidate the ADL checks into a single file.
// Because this is a variable and not a function, it's guaranteed that ADL won't be used. However,
// implementations are allowed to use a different mechanism to achieve this effect, so this check is
// libc++-specific.
LIBCPP_STATIC_ASSERT(std::is_class_v<decltype(std::ranges::uninitialized_copy)>);

static_assert(std::is_invocable_v<decltype(std::ranges::uninitialized_copy), int*, int*, long*, long*>);
struct NotConvertibleFromInt {};
static_assert(!std::is_invocable_v<decltype(std::ranges::uninitialized_copy), int*, int*, NotConvertibleFromInt*,
                                   NotConvertibleFromInt*>);

int main(int, char**) {
  // An empty range -- no default constructors should be invoked.
  {
    Counted in[] = {Counted()};
    Buffer<Counted, 1> out;
    Counted::reset();

    {
      auto result = std::ranges::uninitialized_copy(in, in, out.begin(), out.end());
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == in);
      assert(result.out == out.begin());
    }

    {
      std::ranges::empty_view<Counted> view;
      auto result = std::ranges::uninitialized_copy(view, out);
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == view.begin());
      assert(result.out == out.begin());
    }

    {
      forward_iterator<Counted*> it(in);
      std::ranges::subrange range(it, sentinel_wrapper<forward_iterator<Counted*>>(it));

      auto result = std::ranges::uninitialized_copy(range.begin(), range.end(), out.begin(), out.end());
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == it);
      assert(result.out == out.begin());
    }

    {
      forward_iterator<Counted*> it(in);
      std::ranges::subrange range(it, sentinel_wrapper<forward_iterator<Counted*>>(it));

      auto result = std::ranges::uninitialized_copy(range, out);
      assert(Counted::current_objects == 0);
      assert(Counted::total_objects == 0);
      assert(Counted::total_copies == 0);
      assert(result.in == it);
      assert(result.out == out.begin());
    }
    Counted::reset();
  }

  // A range containing several objects, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, N> out;
    Counted::reset();

    auto result = std::ranges::uninitialized_copy(in, in + N, out.begin(), out.end());
    ASSERT_SAME_TYPE(decltype(result), std::ranges::uninitialized_copy_result<Counted*, Counted*>);

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_copies == N);
    assert(Counted::total_moves == 0);

    assert(std::equal(in, in + N, out.begin(), out.end()));
    assert(result.in == in + N);
    assert(result.out == out.end());

    std::destroy(out.begin(), out.end());
  }
  Counted::reset();

  // A range containing several objects, (range) overload.
  {
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, N> out;
    Counted::reset();

    std::ranges::subrange range(in, in + N);
    auto result = std::ranges::uninitialized_copy(range, out);
    ASSERT_SAME_TYPE(decltype(result), std::ranges::uninitialized_copy_result<Counted*, Counted*>);

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_copies == N);
    assert(Counted::total_moves == 0);

    assert(std::equal(in, in + N, out.begin(), out.end()));
    assert(result.in == in + N);
    assert(result.out == out.end());

    std::destroy(out.begin(), out.end());
  }
  Counted::reset();

  // Using `counted_iterator`.
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    std::counted_iterator iter(in, N);
    auto result = std::ranges::uninitialized_copy(iter, std::default_sentinel, out.begin(), out.end());

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_copies == N);
    assert(Counted::total_moves == 0);
    assert(std::equal(in, in + N, out.begin(), out.begin() + N));

    assert(result.in == iter + N);
    assert(result.out == out.begin() + N);

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();

  // Using `views::counted`.
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    auto view = std::views::counted(in, N);
    auto result = std::ranges::uninitialized_copy(view, out);

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_copies == N);
    assert(Counted::total_moves == 0);
    assert(std::equal(in, in + N, out.begin(), out.begin() + N));

    assert(result.in == view.begin() + N);
    assert(result.out == out.begin() + N);

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();

  // Using `reverse_view`.
  {
    constexpr int N = 3;
    Counted in[] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, 5> out;
    Counted::reset();

    std::ranges::subrange range(in, in + N);
    auto view = std::ranges::views::reverse(range);
    auto result = std::ranges::uninitialized_copy(view, out);

    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(Counted::total_copies == N);
    assert(Counted::total_moves == 0);

    Counted expected[N] = {Counted(3), Counted(2), Counted(1)};
    assert(std::equal(out.begin(), out.begin() + N, expected, expected + N));

    assert(result.in == view.begin() + N);
    assert(result.out == out.begin() + N);

    std::destroy(out.begin(), out.begin() + N);
  }
  Counted::reset();

  // Any existing values should be overwritten by copy constructors.
  {
    constexpr int N = 5;
    int in[N] = {1, 2, 3, 4, 5};
    int out[N] = {6, 7, 8, 9, 10};
    assert(!std::equal(in, in + N, in, out + N));

    std::ranges::uninitialized_copy(in, in + 1, out, out + N);
    assert(out[0] == 1);
    assert(out[1] == 7);

    std::ranges::uninitialized_copy(in, in + N, out, out + N);
    assert(std::equal(in, in + N, out, out + N));
  }

  // An exception is thrown while objects are being created -- objects not yet overwritten should
  // stay valid. (iterator, sentinel) overload.
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    constexpr int M = 3;
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Counted out[N] = {Counted(6), Counted(7), Counted(8), Counted(9), Counted(10)};
    Counted::reset();

    Counted::throw_on = M; // When constructing out[3].
    try {
      std::ranges::uninitialized_copy(in, in + N, out, out + N);
      assert(false);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == M);
    assert(Counted::total_copies == M);
    assert(Counted::total_moves == 0);

    assert(out[4].value == 10);
  }
  Counted::reset();

  // An exception is thrown while objects are being created -- objects not yet overwritten should
  // stay valid. (range) overload.
  {
    constexpr int M = 3;
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Counted out[N] = {Counted(6), Counted(7), Counted(8), Counted(9), Counted(10)};
    Counted::reset();

    Counted::throw_on = M; // When constructing out[3].
    try {
      std::ranges::uninitialized_copy(in, out);
      assert(false);
    } catch (...) {
    }
    assert(Counted::current_objects == 0);
    assert(Counted::total_objects == M);
    assert(Counted::total_copies == M);
    assert(Counted::total_moves == 0);

    assert(out[4].value == 10);
  }
  Counted::reset();
#endif // TEST_HAS_NO_EXCEPTIONS

  // Works with const iterators, (iter, sentinel) overload.
  {
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, N> out;
    Counted::reset();

    std::ranges::uninitialized_copy(in, in + N, out.cbegin(), out.cend());
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::equal(in, in + N, out.begin(), out.end()));

    std::destroy(out.begin(), out.end());
  }
  Counted::reset();

  // Works with const iterators, (range) overload.
  {
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, N> out;
    Counted::reset();

    std::ranges::subrange out_range(out.cbegin(), out.cend());
    std::ranges::uninitialized_copy(in, out_range);
    assert(Counted::current_objects == N);
    assert(Counted::total_objects == N);
    assert(std::equal(in, in + N, out.begin(), out.end()));

    std::destroy(out.begin(), out.end());
  }
  Counted::reset();

  // Conversions, (iter, sentinel) overload.
  {
    constexpr int N = 3;
    double in[N] = {1.0, 2.0, 3.0};
    Buffer<int, N> out;

    std::ranges::uninitialized_copy(in, in + N, out.begin(), out.end());
    assert(std::equal(in, in + N, out.begin(), out.end()));
  }

  // Conversions, (range) overload.
  {
    constexpr int N = 3;
    double in[N] = {1.0, 2.0, 3.0};
    Buffer<int, N> out;

    std::ranges::uninitialized_copy(in, out);
    assert(std::equal(in, in + N, out.begin(), out.end()));
  }

  // Destination range is shorter than the source range, (iter, sentinel) overload.
  {
    constexpr int M = 3;
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, M> out;
    Counted::reset();

    auto result = std::ranges::uninitialized_copy(in, in + N, out.begin(), out.end());
    assert(Counted::current_objects == M);
    assert(Counted::total_objects == M);
    assert(Counted::total_copies == M);
    assert(Counted::total_moves == 0);

    assert(std::equal(in, in + M, out.begin(), out.end()));
    assert(result.in == in + M);
    assert(result.out == out.end());
  }

  // Destination range is shorter than the source range, (range) overload.
  {
    constexpr int M = 3;
    constexpr int N = 5;
    Counted in[N] = {Counted(1), Counted(2), Counted(3), Counted(4), Counted(5)};
    Buffer<Counted, M> out;
    Counted::reset();

    std::ranges::subrange range(in, in + N);
    auto result = std::ranges::uninitialized_copy(range, out);
    assert(Counted::current_objects == M);
    assert(Counted::total_objects == M);
    assert(Counted::total_copies == M);
    assert(Counted::total_moves == 0);

    assert(std::equal(in, in + M, out.begin(), out.end()));
    assert(result.in == in + M);
    assert(result.out == out.end());
  }

  return 0;
}
