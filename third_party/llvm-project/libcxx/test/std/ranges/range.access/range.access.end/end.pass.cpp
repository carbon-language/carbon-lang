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

// std::ranges::end

#include <ranges>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"

using RangeEndT = decltype(std::ranges::end)&;
using RangeCEndT = decltype(std::ranges::cend)&;

static int globalBuff[8];

static_assert(!std::is_invocable_v<RangeEndT, int (&&)[]>);
static_assert(!std::is_invocable_v<RangeEndT, int (&)[]>);
static_assert(!std::is_invocable_v<RangeEndT, int (&&)[10]>);
static_assert( std::is_invocable_v<RangeEndT, int (&)[10]>);

struct EndMember {
  int x;
  constexpr const int *begin() const { return nullptr; }
  constexpr const int *end() const { return &x; }
};

// Ensure that we can't call with rvalues with borrowing disabled.
static_assert( std::is_invocable_v<RangeEndT,  EndMember &>);
static_assert( std::is_invocable_v<RangeEndT,  EndMember const&>);
static_assert(!std::is_invocable_v<RangeEndT,  EndMember &&>);
static_assert( std::is_invocable_v<RangeCEndT, EndMember &>);
static_assert( std::is_invocable_v<RangeCEndT, EndMember const&>);

constexpr bool testArray() {
  int a[2];
  assert(std::ranges::end(a) == a + 2);
  assert(std::ranges::cend(a) == a + 2);

  int b[2][2];
  assert(std::ranges::end(b) == b + 2);
  assert(std::ranges::cend(b) == b + 2);

  EndMember c[2];
  assert(std::ranges::end(c) == c + 2);
  assert(std::ranges::cend(c) == c + 2);

  return true;
}

struct EndMemberFunction {
  int x;
  constexpr const int *begin() const { return nullptr; }
  constexpr const int *end() const { return &x; }
  friend constexpr int *end(EndMemberFunction const&);
};

struct EndMemberReturnsInt {
  int begin() const;
  int end() const;
};

static_assert(!std::is_invocable_v<RangeEndT, EndMemberReturnsInt const&>);

struct EndMemberReturnsVoidPtr {
  const void *begin() const;
  const void *end() const;
};

static_assert(!std::is_invocable_v<RangeEndT, EndMemberReturnsVoidPtr const&>);

struct Empty { };
struct EmptyEndMember {
  Empty begin() const;
  Empty end() const;
};
struct EmptyPtrEndMember {
  Empty x;
  constexpr const Empty *begin() const { return nullptr; }
  constexpr const Empty *end() const { return &x; }
};

static_assert(!std::is_invocable_v<RangeEndT, EmptyEndMember const&>);

struct PtrConvertible {
  operator int*() const;
};
struct PtrConvertibleEndMember {
  PtrConvertible begin() const;
  PtrConvertible end() const;
};

static_assert(!std::is_invocable_v<RangeEndT, PtrConvertibleEndMember const&>);

struct NoBeginMember {
  constexpr const int *end();
};

static_assert(!std::is_invocable_v<RangeEndT, NoBeginMember const&>);

struct NonConstEndMember {
  int x;
  constexpr int *begin() { return nullptr; }
  constexpr int *end() { return &x; }
};

static_assert( std::is_invocable_v<RangeEndT,  NonConstEndMember &>);
static_assert(!std::is_invocable_v<RangeEndT,  NonConstEndMember const&>);
static_assert(!std::is_invocable_v<RangeCEndT, NonConstEndMember &>);
static_assert(!std::is_invocable_v<RangeCEndT, NonConstEndMember const&>);

struct EnabledBorrowingEndMember {
  constexpr int *begin() const { return nullptr; }
  constexpr int *end() const { return &globalBuff[0]; }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<EnabledBorrowingEndMember> = true;

constexpr bool testEndMember() {
  EndMember a;
  assert(std::ranges::end(a) == &a.x);
  assert(std::ranges::cend(a) == &a.x);

  NonConstEndMember b;
  assert(std::ranges::end(b) == &b.x);

  EnabledBorrowingEndMember c;
  assert(std::ranges::end(std::move(c)) == &globalBuff[0]);

  EndMemberFunction d;
  assert(std::ranges::end(d) == &d.x);
  assert(std::ranges::cend(d) == &d.x);

  EmptyPtrEndMember e;
  assert(std::ranges::end(e) == &e.x);
  assert(std::ranges::cend(e) == &e.x);

  return true;
}

struct EndFunction {
  int x;
  friend constexpr const int *begin(EndFunction const&) { return nullptr; }
  friend constexpr const int *end(EndFunction const& bf) { return &bf.x; }
};

static_assert( std::is_invocable_v<RangeEndT, EndFunction const&>);
static_assert(!std::is_invocable_v<RangeEndT, EndFunction &&>);

static_assert( std::is_invocable_v<RangeEndT,  EndFunction const&>);
static_assert(!std::is_invocable_v<RangeEndT,  EndFunction &&>);
static_assert(!std::is_invocable_v<RangeEndT,  EndFunction &>);
static_assert( std::is_invocable_v<RangeCEndT, EndFunction const&>);
static_assert( std::is_invocable_v<RangeCEndT, EndFunction &>);

struct EndFunctionWithDataMember {
  int x;
  int end;
  friend constexpr const int *begin(EndFunctionWithDataMember const&) { return nullptr; }
  friend constexpr const int *end(EndFunctionWithDataMember const& bf) { return &bf.x; }
};

struct EndFunctionWithPrivateEndMember : private EndMember {
  int y;
  friend constexpr const int *begin(EndFunctionWithPrivateEndMember const&) { return nullptr; }
  friend constexpr const int *end(EndFunctionWithPrivateEndMember const& bf) { return &bf.y; }
};

struct EndFunctionReturnsEmptyPtr {
  Empty x;
  friend constexpr const Empty *begin(EndFunctionReturnsEmptyPtr const&) { return nullptr; }
  friend constexpr const Empty *end(EndFunctionReturnsEmptyPtr const& bf) { return &bf.x; }
};

struct EndFunctionByValue {
  friend constexpr int *begin(EndFunctionByValue) { return nullptr; }
  friend constexpr int *end(EndFunctionByValue) { return &globalBuff[1]; }
};

static_assert(!std::is_invocable_v<RangeCEndT, EndFunctionByValue>);

struct EndFunctionEnabledBorrowing {
  friend constexpr int *begin(EndFunctionEnabledBorrowing) { return nullptr; }
  friend constexpr int *end(EndFunctionEnabledBorrowing) { return &globalBuff[2]; }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<EndFunctionEnabledBorrowing> = true;

struct EndFunctionReturnsInt {
  friend constexpr int begin(EndFunctionReturnsInt const&);
  friend constexpr int end(EndFunctionReturnsInt const&);
};

static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsInt const&>);

struct EndFunctionReturnsVoidPtr {
  friend constexpr void *begin(EndFunctionReturnsVoidPtr const&);
  friend constexpr void *end(EndFunctionReturnsVoidPtr const&);
};

static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsVoidPtr const&>);

struct EndFunctionReturnsEmpty {
  friend constexpr Empty begin(EndFunctionReturnsEmpty const&);
  friend constexpr Empty end(EndFunctionReturnsEmpty const&);
};

static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsEmpty const&>);

struct EndFunctionReturnsPtrConvertible {
  friend constexpr PtrConvertible begin(EndFunctionReturnsPtrConvertible const&);
  friend constexpr PtrConvertible end(EndFunctionReturnsPtrConvertible const&);
};

static_assert(!std::is_invocable_v<RangeEndT, EndFunctionReturnsPtrConvertible const&>);

struct NoBeginFunction {
  friend constexpr const int *end(NoBeginFunction const&);
};

static_assert(!std::is_invocable_v<RangeEndT, NoBeginFunction const&>);

struct BeginMemberEndFunction {
  int x;
  constexpr const int *begin() const { return nullptr; }
  friend constexpr const int *end(BeginMemberEndFunction const& bf) { return &bf.x; }
};

constexpr bool testEndFunction() {
  const EndFunction a{};
  assert(std::ranges::end(a) == &a.x);
  EndFunction aa{};
  assert(std::ranges::cend(aa) == &aa.x);

  EndFunctionByValue b;
  assert(std::ranges::end(b) == &globalBuff[1]);
  assert(std::ranges::cend(b) == &globalBuff[1]);

  EndFunctionEnabledBorrowing c;
  assert(std::ranges::end(std::move(c)) == &globalBuff[2]);

  const EndFunctionReturnsEmptyPtr d{};
  assert(std::ranges::end(d) == &d.x);
  EndFunctionReturnsEmptyPtr dd{};
  assert(std::ranges::cend(dd) == &dd.x);

  const EndFunctionWithDataMember e{};
  assert(std::ranges::end(e) == &e.x);
  EndFunctionWithDataMember ee{};
  assert(std::ranges::cend(ee) == &ee.x);

  const EndFunctionWithPrivateEndMember f{};
  assert(std::ranges::end(f) == &f.y);
  EndFunctionWithPrivateEndMember ff{};
  assert(std::ranges::cend(ff) == &ff.y);

  const BeginMemberEndFunction g{};
  assert(std::ranges::end(g) == &g.x);
  BeginMemberEndFunction gg{};
  assert(std::ranges::cend(gg) == &gg.x);

  return true;
}


ASSERT_NOEXCEPT(std::ranges::end(std::declval<int (&)[10]>()));
ASSERT_NOEXCEPT(std::ranges::cend(std::declval<int (&)[10]>()));

template<class T>
struct NoThrowMemberEnd {
  T begin() const;
  T end() const noexcept;
};
ASSERT_NOEXCEPT(std::ranges::end(std::declval<NoThrowMemberEnd<int*>&>()));
ASSERT_NOEXCEPT(std::ranges::cend(std::declval<NoThrowMemberEnd<int*>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::end(std::declval<NoThrowMemberEnd<ThrowingIterator<int>>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::cend(std::declval<NoThrowMemberEnd<ThrowingIterator<int>>&>()));

template<class T>
struct NoThrowADLEnd {
  T begin() const;
  friend T end(NoThrowADLEnd&) noexcept { return T{}; }
  friend T end(NoThrowADLEnd const&) noexcept { return T{}; }
};
ASSERT_NOEXCEPT(std::ranges::end(std::declval<NoThrowADLEnd<int*>&>()));
ASSERT_NOEXCEPT(std::ranges::cend(std::declval<NoThrowADLEnd<int*>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::end(std::declval<NoThrowADLEnd<ThrowingIterator<int>>&>()));
ASSERT_NOT_NOEXCEPT(std::ranges::cend(std::declval<NoThrowADLEnd<ThrowingIterator<int>>&>()));


int main(int, char**) {
  testArray();
  static_assert(testArray());

  testEndMember();
  static_assert(testEndMember());

  testEndFunction();
  static_assert(testEndFunction());

  return 0;
}
