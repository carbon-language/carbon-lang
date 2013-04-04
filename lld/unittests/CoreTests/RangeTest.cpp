//===- lld/unittest/RangeTest.cpp -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief range.h unit tests.
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lld/Core/range.h"

#include <assert.h>
#include <array>
#include <deque>
#include <forward_list>
#include <iterator>
#include <list>
#include <numeric>
#include <sstream>
#include <vector>

template <typename T, typename U> struct AssertTypesSame;
template <typename T> struct AssertTypesSame<T, T> {};
#define ASSERT_TYPES_SAME(T, U) AssertTypesSame<T, U>()

struct no_begin {};
struct member_begin {
  int *begin();
};
struct free_begin {};
int *begin(free_begin);

template <typename T>
auto type_of_forward(T &&t) -> decltype(std::forward<T>(t)) {
  return std::forward<T>(t);
}

template <typename To> To implicit_cast(To val) { return val; }

void test_traits() {
  using namespace lld::detail;
  ASSERT_TYPES_SAME(begin_result<no_begin>::type, undefined);
  // This causes clang to segfault.
#if 0
  ASSERT_TYPES_SAME(
      begin_result<decltype(type_of_forward(member_begin()))>::type, int *);
#endif
  ASSERT_TYPES_SAME(begin_result<free_begin>::type, int *);
}

TEST(Range, constructors) {
  std::vector<int> v(5);
  std::iota(v.begin(), v.end(), 0);
  lld::range<std::vector<int>::iterator> r = v;
  EXPECT_EQ(v.begin(), r.begin());
  EXPECT_EQ(v.end(), r.end());

  int arr[] = { 1, 2, 3, 4, 5 };
  std::begin(arr);
  lld::range<int *> r2 = arr;
  EXPECT_EQ(5, r2.back());
}

TEST(Range, conversion_to_pointer_range) {
  std::vector<int> v(5);
  std::iota(v.begin(), v.end(), 0);
  lld::range<int *> r = v;
  EXPECT_EQ(&*v.begin(), r.begin());
  EXPECT_EQ(2, r[2]);
}

template <typename Iter> void takes_range(lld::range<Iter> r) {
  int expected = 0;
  for (int val : r) {
    EXPECT_EQ(expected++, val);
  }
}

void takes_ptr_range(lld::range<const int *> r) {
  int expected = 0;
  for (int val : r) {
    EXPECT_EQ(expected++, val);
  }
}

TEST(Range, passing) {
  using lld::make_range;
  using lld::make_ptr_range;
  std::list<int> l(5);
  std::iota(l.begin(), l.end(), 0);
  takes_range(make_range(l));
  takes_range(make_range(implicit_cast<const std::list<int> &>(l)));
  std::deque<int> d(5);
  std::iota(d.begin(), d.end(), 0);
  takes_range(make_range(d));
  takes_range(make_range(implicit_cast<const std::deque<int> &>(d)));
  std::vector<int> v(5);
  std::iota(v.begin(), v.end(), 0);
  takes_range(make_range(v));
  takes_range(make_range(implicit_cast<const std::vector<int> &>(v)));
  // MSVC Can't compile make_ptr_range.
#ifndef _MSC_VER
  static_assert(
      std::is_same<decltype(make_ptr_range(v)), lld::range<int *> >::value,
      "make_ptr_range should return a range of pointers");
  takes_range(make_ptr_range(v));
  takes_range(make_ptr_range(implicit_cast<const std::vector<int> &>(v)));
#endif
  int arr[] = { 0, 1, 2, 3, 4 };
  takes_range(make_range(arr));
  const int carr[] = { 0, 1, 2, 3, 4 };
  takes_range(make_range(carr));

  takes_ptr_range(v);
  takes_ptr_range(implicit_cast<const std::vector<int> &>(v));
  takes_ptr_range(arr);
  takes_ptr_range(carr);
}

TEST(Range, access) {
  std::array<int, 5> a = { { 1, 2, 3, 4, 5 } };
  lld::range<decltype(a.begin())> r = a;
  EXPECT_EQ(4, r[3]);
  EXPECT_EQ(4, r[-2]);
}

template <bool b> struct CompileAssert;
template <> struct CompileAssert<true> {};

#if __has_feature(cxx_constexpr)
constexpr int arr[] = { 1, 2, 3, 4, 5 };
TEST(Range, constexpr) {
  constexpr lld::range<const int *> r(arr, arr + 5);
  CompileAssert<r.front() == 1>();
  CompileAssert<r.size() == 5>();
  CompileAssert<r[4] == 5>();
}
#endif

template <typename Container> void test_slice() {
  Container cont(10);
  std::iota(cont.begin(), cont.end(), 0);
  lld::range<decltype(cont.begin())> r = cont;

  // One argument.
  EXPECT_EQ(10, r.slice(0).size());
  EXPECT_EQ(8, r.slice(2).size());
  EXPECT_EQ(2, r.slice(2).front());
  EXPECT_EQ(1, r.slice(-1).size());
  EXPECT_EQ(9, r.slice(-1).front());

  // Two positive arguments.
  EXPECT_TRUE(r.slice(5, 2).empty());
  EXPECT_EQ(next(cont.begin(), 5), r.slice(5, 2).begin());
  EXPECT_EQ(1, r.slice(1, 2).size());
  EXPECT_EQ(1, r.slice(1, 2).front());

  // Two negative arguments.
  EXPECT_TRUE(r.slice(-2, -5).empty());
  EXPECT_EQ(next(cont.begin(), 8), r.slice(-2, -5).begin());
  EXPECT_EQ(1, r.slice(-2, -1).size());
  EXPECT_EQ(8, r.slice(-2, -1).front());

  // Positive start, negative stop.
  EXPECT_EQ(1, r.slice(6, -3).size());
  EXPECT_EQ(6, r.slice(6, -3).front());
  EXPECT_TRUE(r.slice(6, -5).empty());
  EXPECT_EQ(next(cont.begin(), 6), r.slice(6, -5).begin());

  // Negative start, positive stop.
  EXPECT_TRUE(r.slice(-3, 6).empty());
  EXPECT_EQ(next(cont.begin(), 7), r.slice(-3, 6).begin());
  EXPECT_EQ(1, r.slice(-5, 6).size());
  EXPECT_EQ(5, r.slice(-5, 6).front());
}

TEST(Range, slice) {
  // -fsanitize=undefined complains about this, but only if optimizations are
  // enabled.
#if 0
  test_slice<std::forward_list<int> >();
#endif
  test_slice<std::list<int> >();
  // gcc doesn't like this.
#if !(defined(__GNUC__) && !defined(__clang__)) || defined(_MSC_VER)
  test_slice<std::deque<int> >();
#endif
}

// This test is flaky and I've yet to pin down why. Changing between
// EXPECT_EQ(1, input.front()) and EXPECT_TRUE(input.front() == 1) makes it work
// with VS 2012 in Debug mode. Clang on Linux seems to fail with -03 and -02 -g
// -fsanitize=undefined.
#if 0
TEST(Range, istream_range) {
  std::istringstream stream("1 2 3 4 5");
  // MSVC interprets input as a function declaration if you don't declare start
  // and instead directly pass std::istream_iterator<int>(stream).
  auto start = std::istream_iterator<int>(stream);
  lld::range<std::istream_iterator<int> > input(
      start, std::istream_iterator<int>());
  EXPECT_TRUE(input.front() == 1);
  input.pop_front();
  EXPECT_TRUE(input.front() == 2);
  input.pop_front(2);
  EXPECT_TRUE(input.front() == 4);
  input.pop_front_upto(7);
  EXPECT_TRUE(input.empty());
}
#endif

//! [algorithm using range]
template <typename T> void partial_sum(T &container) {
  using lld::make_range;
  auto range = make_range(container);
  typename T::value_type sum = 0;
  // One would actually use a range-based for loop
  // in this case, but you get the idea:
  for (; !range.empty(); range.pop_front()) {
    sum += range.front();
    range.front() = sum;
  }
}

TEST(Range, user1) {
  std::vector<int> v(5, 2);
  partial_sum(v);
  EXPECT_EQ(8, v[3]);
}
//! [algorithm using range]

//! [algorithm using ptr_range]
void my_write(int fd, lld::range<const char *> buffer) {}

TEST(Range, user2) {
  std::string s("Hello world");
  my_write(1, s);
}
