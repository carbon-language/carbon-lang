//===- STLExtrasTest.cpp - Unit tests for STL extras ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <list>
#include <vector>

using namespace llvm;

namespace {

int f(rank<0>) { return 0; }
int f(rank<1>) { return 1; }
int f(rank<2>) { return 2; }
int f(rank<4>) { return 4; }

TEST(STLExtrasTest, Rank) {
  // We shouldn't get ambiguities and should select the overload of the same
  // rank as the argument.
  EXPECT_EQ(0, f(rank<0>()));
  EXPECT_EQ(1, f(rank<1>()));
  EXPECT_EQ(2, f(rank<2>()));

  // This overload is missing so we end up back at 2.
  EXPECT_EQ(2, f(rank<3>()));

  // But going past 3 should work fine.
  EXPECT_EQ(4, f(rank<4>()));

  // And we can even go higher and just fall back to the last overload.
  EXPECT_EQ(4, f(rank<5>()));
  EXPECT_EQ(4, f(rank<6>()));
}

TEST(STLExtrasTest, EnumerateLValue) {
  // Test that a simple LValue can be enumerated and gives correct results with
  // multiple types, including the empty container.
  std::vector<char> foo = {'a', 'b', 'c'};
  typedef std::pair<std::size_t, char> CharPairType;
  std::vector<CharPairType> CharResults;

  for (auto X : llvm::enumerate(foo)) {
    CharResults.emplace_back(X.index(), X.value());
  }
  ASSERT_EQ(3u, CharResults.size());
  EXPECT_EQ(CharPairType(0u, 'a'), CharResults[0]);
  EXPECT_EQ(CharPairType(1u, 'b'), CharResults[1]);
  EXPECT_EQ(CharPairType(2u, 'c'), CharResults[2]);

  // Test a const range of a different type.
  typedef std::pair<std::size_t, int> IntPairType;
  std::vector<IntPairType> IntResults;
  const std::vector<int> bar = {1, 2, 3};
  for (auto X : llvm::enumerate(bar)) {
    IntResults.emplace_back(X.index(), X.value());
  }
  ASSERT_EQ(3u, IntResults.size());
  EXPECT_EQ(IntPairType(0u, 1), IntResults[0]);
  EXPECT_EQ(IntPairType(1u, 2), IntResults[1]);
  EXPECT_EQ(IntPairType(2u, 3), IntResults[2]);

  // Test an empty range.
  IntResults.clear();
  const std::vector<int> baz{};
  for (auto X : llvm::enumerate(baz)) {
    IntResults.emplace_back(X.index(), X.value());
  }
  EXPECT_TRUE(IntResults.empty());
}

TEST(STLExtrasTest, EnumerateModifyLValue) {
  // Test that you can modify the underlying entries of an lvalue range through
  // the enumeration iterator.
  std::vector<char> foo = {'a', 'b', 'c'};

  for (auto X : llvm::enumerate(foo)) {
    ++X.value();
  }
  EXPECT_EQ('b', foo[0]);
  EXPECT_EQ('c', foo[1]);
  EXPECT_EQ('d', foo[2]);
}

TEST(STLExtrasTest, EnumerateRValueRef) {
  // Test that an rvalue can be enumerated.
  typedef std::pair<std::size_t, int> PairType;
  std::vector<PairType> Results;

  auto Enumerator = llvm::enumerate(std::vector<int>{1, 2, 3});

  for (auto X : llvm::enumerate(std::vector<int>{1, 2, 3})) {
    Results.emplace_back(X.index(), X.value());
  }

  ASSERT_EQ(3u, Results.size());
  EXPECT_EQ(PairType(0u, 1), Results[0]);
  EXPECT_EQ(PairType(1u, 2), Results[1]);
  EXPECT_EQ(PairType(2u, 3), Results[2]);
}

TEST(STLExtrasTest, EnumerateModifyRValue) {
  // Test that when enumerating an rvalue, modification still works (even if
  // this isn't terribly useful, it at least shows that we haven't snuck an
  // extra const in there somewhere.
  typedef std::pair<std::size_t, char> PairType;
  std::vector<PairType> Results;

  for (auto X : llvm::enumerate(std::vector<char>{'1', '2', '3'})) {
    ++X.value();
    Results.emplace_back(X.index(), X.value());
  }

  ASSERT_EQ(3u, Results.size());
  EXPECT_EQ(PairType(0u, '2'), Results[0]);
  EXPECT_EQ(PairType(1u, '3'), Results[1]);
  EXPECT_EQ(PairType(2u, '4'), Results[2]);
}

template <bool B> struct CanMove {};
template <> struct CanMove<false> {
  CanMove(CanMove &&) = delete;

  CanMove() = default;
  CanMove(const CanMove &) = default;
};

template <bool B> struct CanCopy {};
template <> struct CanCopy<false> {
  CanCopy(const CanCopy &) = delete;

  CanCopy() = default;
  CanCopy(CanCopy &&) = default;
};

template <bool Moveable, bool Copyable>
class Counted : CanMove<Moveable>, CanCopy<Copyable> {
  int &C;
  int &M;
  int &D;

public:
  explicit Counted(int &C, int &M, int &D) : C(C), M(M), D(D) {}
  Counted(const Counted &O) : CanCopy<Copyable>(O), C(O.C), M(O.M), D(O.D) {
    ++C;
  }
  Counted(Counted &&O)
      : CanMove<Moveable>(std::move(O)), C(O.C), M(O.M), D(O.D) {
    ++M;
  }
  ~Counted() { ++D; }
};

template <bool Moveable, bool Copyable>
struct Range : Counted<Moveable, Copyable> {
  using Counted<Moveable, Copyable>::Counted;
  int *begin() { return nullptr; }
  int *end() { return nullptr; }
};

TEST(STLExtrasTest, EnumerateLifetimeSemanticsPRValue) {
  int Copies = 0;
  int Moves = 0;
  int Destructors = 0;
  {
    auto E = enumerate(Range<true, false>(Copies, Moves, Destructors));
    (void)E;
    // Doesn't compile.  rvalue ranges must be moveable.
    // auto E2 = enumerate(Range<false, true>(Copies, Moves, Destructors));
    EXPECT_EQ(0, Copies);
    EXPECT_EQ(1, Moves);
    EXPECT_EQ(1, Destructors);
  }
  EXPECT_EQ(0, Copies);
  EXPECT_EQ(1, Moves);
  EXPECT_EQ(2, Destructors);
}

TEST(STLExtrasTest, EnumerateLifetimeSemanticsRValue) {
  // With an rvalue, it should not be destroyed until the end of the scope.
  int Copies = 0;
  int Moves = 0;
  int Destructors = 0;
  {
    Range<true, false> R(Copies, Moves, Destructors);
    {
      auto E = enumerate(std::move(R));
      (void)E;
      // Doesn't compile.  rvalue ranges must be moveable.
      // auto E2 = enumerate(Range<false, true>(Copies, Moves, Destructors));
      EXPECT_EQ(0, Copies);
      EXPECT_EQ(1, Moves);
      EXPECT_EQ(0, Destructors);
    }
    EXPECT_EQ(0, Copies);
    EXPECT_EQ(1, Moves);
    EXPECT_EQ(1, Destructors);
  }
  EXPECT_EQ(0, Copies);
  EXPECT_EQ(1, Moves);
  EXPECT_EQ(2, Destructors);
}

TEST(STLExtrasTest, EnumerateLifetimeSemanticsLValue) {
  // With an lvalue, it should not be destroyed even after the end of the scope.
  // lvalue ranges need be neither copyable nor moveable.
  int Copies = 0;
  int Moves = 0;
  int Destructors = 0;
  {
    Range<false, false> R(Copies, Moves, Destructors);
    {
      auto E = enumerate(R);
      (void)E;
      EXPECT_EQ(0, Copies);
      EXPECT_EQ(0, Moves);
      EXPECT_EQ(0, Destructors);
    }
    EXPECT_EQ(0, Copies);
    EXPECT_EQ(0, Moves);
    EXPECT_EQ(0, Destructors);
  }
  EXPECT_EQ(0, Copies);
  EXPECT_EQ(0, Moves);
  EXPECT_EQ(1, Destructors);
}

TEST(STLExtrasTest, ApplyTuple) {
  auto T = std::make_tuple(1, 3, 7);
  auto U = llvm::apply_tuple(
      [](int A, int B, int C) { return std::make_tuple(A - B, B - C, C - A); },
      T);

  EXPECT_EQ(-2, std::get<0>(U));
  EXPECT_EQ(-4, std::get<1>(U));
  EXPECT_EQ(6, std::get<2>(U));

  auto V = llvm::apply_tuple(
      [](int A, int B, int C) {
        return std::make_tuple(std::make_pair(A, char('A' + A)),
                               std::make_pair(B, char('A' + B)),
                               std::make_pair(C, char('A' + C)));
      },
      T);

  EXPECT_EQ(std::make_pair(1, 'B'), std::get<0>(V));
  EXPECT_EQ(std::make_pair(3, 'D'), std::get<1>(V));
  EXPECT_EQ(std::make_pair(7, 'H'), std::get<2>(V));
}

class apply_variadic {
  static int apply_one(int X) { return X + 1; }
  static char apply_one(char C) { return C + 1; }
  static StringRef apply_one(StringRef S) { return S.drop_back(); }

public:
  template <typename... Ts> auto operator()(Ts &&... Items) {
    return std::make_tuple(apply_one(Items)...);
  }
};

TEST(STLExtrasTest, ApplyTupleVariadic) {
  auto Items = std::make_tuple(1, llvm::StringRef("Test"), 'X');
  auto Values = apply_tuple(apply_variadic(), Items);

  EXPECT_EQ(2, std::get<0>(Values));
  EXPECT_EQ("Tes", std::get<1>(Values));
  EXPECT_EQ('Y', std::get<2>(Values));
}

TEST(STLExtrasTest, CountAdaptor) {
  std::vector<int> v;

  v.push_back(1);
  v.push_back(2);
  v.push_back(1);
  v.push_back(4);
  v.push_back(3);
  v.push_back(2);
  v.push_back(1);

  EXPECT_EQ(3, count(v, 1));
  EXPECT_EQ(2, count(v, 2));
  EXPECT_EQ(1, count(v, 3));
  EXPECT_EQ(1, count(v, 4));
}

TEST(STLExtrasTest, for_each) {
  std::vector<int> v{0, 1, 2, 3, 4};
  int count = 0;

  llvm::for_each(v, [&count](int) { ++count; });
  EXPECT_EQ(5, count);
}

TEST(STLExtrasTest, ToVector) {
  std::vector<char> v = {'a', 'b', 'c'};
  auto Enumerated = to_vector<4>(enumerate(v));
  ASSERT_EQ(3u, Enumerated.size());
  for (size_t I = 0; I < v.size(); ++I) {
    EXPECT_EQ(I, Enumerated[I].index());
    EXPECT_EQ(v[I], Enumerated[I].value());
  }

  auto EnumeratedImplicitSize = to_vector(enumerate(v));
  ASSERT_EQ(3u, EnumeratedImplicitSize.size());
  for (size_t I = 0; I < v.size(); ++I) {
    EXPECT_EQ(I, EnumeratedImplicitSize[I].index());
    EXPECT_EQ(v[I], EnumeratedImplicitSize[I].value());
  }
}

TEST(STLExtrasTest, ConcatRange) {
  std::vector<int> Expected = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> Test;

  std::vector<int> V1234 = {1, 2, 3, 4};
  std::list<int> L56 = {5, 6};
  SmallVector<int, 2> SV78 = {7, 8};

  // Use concat across different sized ranges of different types with different
  // iterators.
  for (int &i : concat<int>(V1234, L56, SV78))
    Test.push_back(i);
  EXPECT_EQ(Expected, Test);

  // Use concat between a temporary, an L-value, and an R-value to make sure
  // complex lifetimes work well.
  Test.clear();
  for (int &i : concat<int>(std::vector<int>(V1234), L56, std::move(SV78)))
    Test.push_back(i);
  EXPECT_EQ(Expected, Test);
}

TEST(STLExtrasTest, PartitionAdaptor) {
  std::vector<int> V = {1, 2, 3, 4, 5, 6, 7, 8};

  auto I = partition(V, [](int i) { return i % 2 == 0; });
  ASSERT_EQ(V.begin() + 4, I);

  // Sort the two halves as partition may have messed with the order.
  llvm::sort(V.begin(), I);
  llvm::sort(I, V.end());

  EXPECT_EQ(2, V[0]);
  EXPECT_EQ(4, V[1]);
  EXPECT_EQ(6, V[2]);
  EXPECT_EQ(8, V[3]);
  EXPECT_EQ(1, V[4]);
  EXPECT_EQ(3, V[5]);
  EXPECT_EQ(5, V[6]);
  EXPECT_EQ(7, V[7]);
}

TEST(STLExtrasTest, EraseIf) {
  std::vector<int> V = {1, 2, 3, 4, 5, 6, 7, 8};

  erase_if(V, [](int i) { return i % 2 == 0; });
  EXPECT_EQ(4u, V.size());
  EXPECT_EQ(1, V[0]);
  EXPECT_EQ(3, V[1]);
  EXPECT_EQ(5, V[2]);
  EXPECT_EQ(7, V[3]);
}

TEST(STLExtrasTest, AppendRange) {
  auto AppendVals = {3};
  std::vector<int> V = {1, 2};
  append_range(V, AppendVals);
  EXPECT_EQ(1, V[0]);
  EXPECT_EQ(2, V[1]);
  EXPECT_EQ(3, V[2]);
}

namespace some_namespace {
struct some_struct {
  std::vector<int> data;
  std::string swap_val;
};

std::vector<int>::const_iterator begin(const some_struct &s) {
  return s.data.begin();
}

std::vector<int>::const_iterator end(const some_struct &s) {
  return s.data.end();
}

void swap(some_struct &lhs, some_struct &rhs) {
  // make swap visible as non-adl swap would even seem to
  // work with std::swap which defaults to moving
  lhs.swap_val = "lhs";
  rhs.swap_val = "rhs";
}
} // namespace some_namespace

TEST(STLExtrasTest, ADLTest) {
  some_namespace::some_struct s{{1, 2, 3, 4, 5}, ""};
  some_namespace::some_struct s2{{2, 4, 6, 8, 10}, ""};

  EXPECT_EQ(*adl_begin(s), 1);
  EXPECT_EQ(*(adl_end(s) - 1), 5);

  adl_swap(s, s2);
  EXPECT_EQ(s.swap_val, "lhs");
  EXPECT_EQ(s2.swap_val, "rhs");

  int count = 0;
  llvm::for_each(s, [&count](int) { ++count; });
  EXPECT_EQ(5, count);
}

TEST(STLExtrasTest, EmptyTest) {
  std::vector<void*> V;
  EXPECT_TRUE(llvm::empty(V));
  V.push_back(nullptr);
  EXPECT_FALSE(llvm::empty(V));

  std::initializer_list<int> E = {};
  std::initializer_list<int> NotE = {7, 13, 42};
  EXPECT_TRUE(llvm::empty(E));
  EXPECT_FALSE(llvm::empty(NotE));

  auto R0 = make_range(V.begin(), V.begin());
  EXPECT_TRUE(llvm::empty(R0));
  auto R1 = make_range(V.begin(), V.end());
  EXPECT_FALSE(llvm::empty(R1));
}

TEST(STLExtrasTest, DropBeginTest) {
  SmallVector<int, 5> vec{0, 1, 2, 3, 4};

  for (int n = 0; n < 5; ++n) {
    int i = n;
    for (auto &v : drop_begin(vec, n)) {
      EXPECT_EQ(v, i);
      i += 1;
    }
    EXPECT_EQ(i, 5);
  }
}

TEST(STLExtrasTest, DropBeginDefaultTest) {
  SmallVector<int, 5> vec{0, 1, 2, 3, 4};

  int i = 1;
  for (auto &v : drop_begin(vec)) {
    EXPECT_EQ(v, i);
    i += 1;
  }
  EXPECT_EQ(i, 5);
}

TEST(STLExtrasTest, EarlyIncrementTest) {
  std::list<int> L = {1, 2, 3, 4};

  auto EIR = make_early_inc_range(L);

  auto I = EIR.begin();
  auto EI = EIR.end();
  EXPECT_NE(I, EI);

  EXPECT_EQ(1, *I);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#ifndef NDEBUG
  // Repeated dereferences are not allowed.
  EXPECT_DEATH(*I, "Cannot dereference");
  // Comparison after dereference is not allowed.
  EXPECT_DEATH((void)(I == EI), "Cannot compare");
  EXPECT_DEATH((void)(I != EI), "Cannot compare");
#endif
#endif

  ++I;
  EXPECT_NE(I, EI);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#ifndef NDEBUG
  // You cannot increment prior to dereference.
  EXPECT_DEATH(++I, "Cannot increment");
#endif
#endif
  EXPECT_EQ(2, *I);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#ifndef NDEBUG
  // Repeated dereferences are not allowed.
  EXPECT_DEATH(*I, "Cannot dereference");
#endif
#endif

  // Inserting shouldn't break anything. We should be able to keep dereferencing
  // the currrent iterator and increment. The increment to go to the "next"
  // iterator from before we inserted.
  L.insert(std::next(L.begin(), 2), -1);
  ++I;
  EXPECT_EQ(3, *I);

  // Erasing the front including the current doesn't break incrementing.
  L.erase(L.begin(), std::prev(L.end()));
  ++I;
  EXPECT_EQ(4, *I);
  ++I;
  EXPECT_EQ(EIR.end(), I);
}

// A custom iterator that returns a pointer when dereferenced. This is used to
// test make_early_inc_range with iterators that do not return a reference on
// dereferencing.
struct CustomPointerIterator
    : public iterator_adaptor_base<CustomPointerIterator,
                                   std::list<int>::iterator,
                                   std::forward_iterator_tag> {
  using base_type =
      iterator_adaptor_base<CustomPointerIterator, std::list<int>::iterator,
                            std::forward_iterator_tag>;

  explicit CustomPointerIterator(std::list<int>::iterator I) : base_type(I) {}

  // Retrieve a pointer to the current int.
  int *operator*() const { return &*base_type::wrapped(); }
};

// Make sure make_early_inc_range works with iterators that do not return a
// reference on dereferencing. The test is similar to EarlyIncrementTest, but
// uses CustomPointerIterator.
TEST(STLExtrasTest, EarlyIncrementTestCustomPointerIterator) {
  std::list<int> L = {1, 2, 3, 4};

  auto CustomRange = make_range(CustomPointerIterator(L.begin()),
                                CustomPointerIterator(L.end()));
  auto EIR = make_early_inc_range(CustomRange);

  auto I = EIR.begin();
  auto EI = EIR.end();
  EXPECT_NE(I, EI);

  EXPECT_EQ(&*L.begin(), *I);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#ifndef NDEBUG
  // Repeated dereferences are not allowed.
  EXPECT_DEATH(*I, "Cannot dereference");
  // Comparison after dereference is not allowed.
  EXPECT_DEATH((void)(I == EI), "Cannot compare");
  EXPECT_DEATH((void)(I != EI), "Cannot compare");
#endif
#endif

  ++I;
  EXPECT_NE(I, EI);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#ifndef NDEBUG
  // You cannot increment prior to dereference.
  EXPECT_DEATH(++I, "Cannot increment");
#endif
#endif
  EXPECT_EQ(&*std::next(L.begin()), *I);
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#ifndef NDEBUG
  // Repeated dereferences are not allowed.
  EXPECT_DEATH(*I, "Cannot dereference");
#endif
#endif

  // Inserting shouldn't break anything. We should be able to keep dereferencing
  // the currrent iterator and increment. The increment to go to the "next"
  // iterator from before we inserted.
  L.insert(std::next(L.begin(), 2), -1);
  ++I;
  EXPECT_EQ(&*std::next(L.begin(), 3), *I);

  // Erasing the front including the current doesn't break incrementing.
  L.erase(L.begin(), std::prev(L.end()));
  ++I;
  EXPECT_EQ(&*L.begin(), *I);
  ++I;
  EXPECT_EQ(EIR.end(), I);
}

TEST(STLExtrasTest, splat) {
  std::vector<int> V;
  EXPECT_FALSE(is_splat(V));

  V.push_back(1);
  EXPECT_TRUE(is_splat(V));

  V.push_back(1);
  V.push_back(1);
  EXPECT_TRUE(is_splat(V));

  V.push_back(2);
  EXPECT_FALSE(is_splat(V));
}

TEST(STLExtrasTest, to_address) {
  int *V1 = new int;
  EXPECT_EQ(V1, to_address(V1));

  // Check fancy pointer overload for unique_ptr
  std::unique_ptr<int> V2 = std::make_unique<int>(0);
  EXPECT_EQ(V2.get(), llvm::to_address(V2));

  V2.reset(V1);
  EXPECT_EQ(V1, llvm::to_address(V2));
  V2.release();

  // Check fancy pointer overload for shared_ptr
  std::shared_ptr<int> V3 = std::make_shared<int>(0);
  std::shared_ptr<int> V4 = V3;
  EXPECT_EQ(V3.get(), V4.get());
  EXPECT_EQ(V3.get(), llvm::to_address(V3));
  EXPECT_EQ(V4.get(), llvm::to_address(V4));

  V3.reset(V1);
  EXPECT_EQ(V1, llvm::to_address(V3));
}

TEST(STLExtrasTest, partition_point) {
  std::vector<int> V = {1, 3, 5, 7, 9};

  // Range version.
  EXPECT_EQ(V.begin() + 3,
            partition_point(V, [](unsigned X) { return X < 7; }));
  EXPECT_EQ(V.begin(), partition_point(V, [](unsigned X) { return X < 1; }));
  EXPECT_EQ(V.end(), partition_point(V, [](unsigned X) { return X < 50; }));
}

TEST(STLExtrasTest, hasSingleElement) {
  const std::vector<int> V0 = {}, V1 = {1}, V2 = {1, 2};
  const std::vector<int> V10(10);

  EXPECT_EQ(hasSingleElement(V0), false);
  EXPECT_EQ(hasSingleElement(V1), true);
  EXPECT_EQ(hasSingleElement(V2), false);
  EXPECT_EQ(hasSingleElement(V10), false);
}

TEST(STLExtrasTest, hasNItems) {
  const std::list<int> V0 = {}, V1 = {1}, V2 = {1, 2};
  const std::list<int> V3 = {1, 3, 5};

  EXPECT_TRUE(hasNItems(V0, 0));
  EXPECT_FALSE(hasNItems(V0, 2));
  EXPECT_TRUE(hasNItems(V1, 1));
  EXPECT_FALSE(hasNItems(V1, 2));

  EXPECT_TRUE(hasNItems(V3.begin(), V3.end(), 3, [](int x) { return x < 10; }));
  EXPECT_TRUE(hasNItems(V3.begin(), V3.end(), 0, [](int x) { return x > 10; }));
  EXPECT_TRUE(hasNItems(V3.begin(), V3.end(), 2, [](int x) { return x < 5; }));
}

TEST(STLExtras, hasNItemsOrMore) {
  const std::list<int> V0 = {}, V1 = {1}, V2 = {1, 2};
  const std::list<int> V3 = {1, 3, 5};

  EXPECT_TRUE(hasNItemsOrMore(V1, 1));
  EXPECT_FALSE(hasNItemsOrMore(V1, 2));

  EXPECT_TRUE(hasNItemsOrMore(V2, 1));
  EXPECT_TRUE(hasNItemsOrMore(V2, 2));
  EXPECT_FALSE(hasNItemsOrMore(V2, 3));

  EXPECT_TRUE(hasNItemsOrMore(V3, 3));
  EXPECT_FALSE(hasNItemsOrMore(V3, 4));

  EXPECT_TRUE(
      hasNItemsOrMore(V3.begin(), V3.end(), 3, [](int x) { return x < 10; }));
  EXPECT_FALSE(
      hasNItemsOrMore(V3.begin(), V3.end(), 3, [](int x) { return x > 10; }));
  EXPECT_TRUE(
      hasNItemsOrMore(V3.begin(), V3.end(), 2, [](int x) { return x < 5; }));
}

TEST(STLExtras, hasNItemsOrLess) {
  const std::list<int> V0 = {}, V1 = {1}, V2 = {1, 2};
  const std::list<int> V3 = {1, 3, 5};

  EXPECT_TRUE(hasNItemsOrLess(V0, 0));
  EXPECT_TRUE(hasNItemsOrLess(V0, 1));
  EXPECT_TRUE(hasNItemsOrLess(V0, 2));

  EXPECT_FALSE(hasNItemsOrLess(V1, 0));
  EXPECT_TRUE(hasNItemsOrLess(V1, 1));
  EXPECT_TRUE(hasNItemsOrLess(V1, 2));

  EXPECT_FALSE(hasNItemsOrLess(V2, 0));
  EXPECT_FALSE(hasNItemsOrLess(V2, 1));
  EXPECT_TRUE(hasNItemsOrLess(V2, 2));
  EXPECT_TRUE(hasNItemsOrLess(V2, 3));

  EXPECT_FALSE(hasNItemsOrLess(V3, 0));
  EXPECT_FALSE(hasNItemsOrLess(V3, 1));
  EXPECT_FALSE(hasNItemsOrLess(V3, 2));
  EXPECT_TRUE(hasNItemsOrLess(V3, 3));
  EXPECT_TRUE(hasNItemsOrLess(V3, 4));

  EXPECT_TRUE(
      hasNItemsOrLess(V3.begin(), V3.end(), 1, [](int x) { return x == 1; }));
  EXPECT_TRUE(
      hasNItemsOrLess(V3.begin(), V3.end(), 2, [](int x) { return x < 5; }));
  EXPECT_TRUE(
      hasNItemsOrLess(V3.begin(), V3.end(), 5, [](int x) { return x < 5; }));
  EXPECT_FALSE(
      hasNItemsOrLess(V3.begin(), V3.end(), 2, [](int x) { return x < 10; }));
}

TEST(STLExtras, MoveRange) {
  class Foo {
    bool A;

  public:
    Foo() : A(true) {}
    Foo(const Foo &) = delete;
    Foo(Foo &&Other) : A(Other.A) { Other.A = false; }
    Foo &operator=(const Foo &) = delete;
    Foo &operator=(Foo &&Other) {
      if (this != &Other) {
        A = Other.A;
        Other.A = false;
      }
      return *this;
    }
    operator bool() const { return A; }
  };
  SmallVector<Foo, 4U> V1, V2, V3, V4;
  auto HasVal = [](const Foo &Item) { return static_cast<bool>(Item); };
  auto Build = [&] {
    SmallVector<Foo, 4U> Foos;
    Foos.resize(4U);
    return Foos;
  };

  V1.resize(4U);
  EXPECT_TRUE(llvm::all_of(V1, HasVal));

  llvm::move(V1, std::back_inserter(V2));

  // Ensure input container is same size, but its contents were moved out.
  EXPECT_EQ(V1.size(), 4U);
  EXPECT_TRUE(llvm::none_of(V1, HasVal));

  // Ensure output container has the contents of the input container.
  EXPECT_EQ(V2.size(), 4U);
  EXPECT_TRUE(llvm::all_of(V2, HasVal));

  llvm::move(std::move(V2), std::back_inserter(V3));

  EXPECT_TRUE(llvm::none_of(V2, HasVal));
  EXPECT_EQ(V3.size(), 4U);
  EXPECT_TRUE(llvm::all_of(V3, HasVal));

  llvm::move(Build(), std::back_inserter(V4));
  EXPECT_EQ(V4.size(), 4U);
  EXPECT_TRUE(llvm::all_of(V4, HasVal));
}

TEST(STLExtras, Unique) {
  std::vector<int> V = {1, 5, 5, 4, 3, 3, 3};

  auto I = llvm::unique(V, [](int a, int b) { return a == b; });

  EXPECT_EQ(I, V.begin() + 4);

  EXPECT_EQ(1, V[0]);
  EXPECT_EQ(5, V[1]);
  EXPECT_EQ(4, V[2]);
  EXPECT_EQ(3, V[3]);
}

TEST(STLExtrasTest, MakeVisitorOneCallable) {
  auto IdentityLambda = [](auto X) { return X; };
  auto IdentityVisitor = makeVisitor(IdentityLambda);
  EXPECT_EQ(IdentityLambda(1), IdentityVisitor(1));
  EXPECT_EQ(IdentityLambda(2.0f), IdentityVisitor(2.0f));
  EXPECT_TRUE((std::is_same<decltype(IdentityLambda(IdentityLambda)),
                            decltype(IdentityLambda)>::value));
  EXPECT_TRUE((std::is_same<decltype(IdentityVisitor(IdentityVisitor)),
                            decltype(IdentityVisitor)>::value));
}

TEST(STLExtrasTest, MakeVisitorTwoCallables) {
  auto Visitor =
      makeVisitor([](int) { return 0; }, [](std::string) { return 1; });
  EXPECT_EQ(Visitor(42), 0);
  EXPECT_EQ(Visitor("foo"), 1);
}

TEST(STLExtrasTest, MakeVisitorCallableMultipleOperands) {
  auto Second = makeVisitor([](int I, float F) { return F; },
                            [](float F, int I) { return I; });
  EXPECT_EQ(Second(1.f, 1), 1);
  EXPECT_EQ(Second(1, 1.f), 1.f);
}

TEST(STLExtrasTest, MakeVisitorDefaultCase) {
  {
    auto Visitor = makeVisitor([](int I) { return I + 100; },
                               [](float F) { return F * 2; },
                               [](auto) { return -1; });
    EXPECT_EQ(Visitor(24), 124);
    EXPECT_EQ(Visitor(2.f), 4.f);
    EXPECT_EQ(Visitor(2.), -1);
    EXPECT_EQ(Visitor(Visitor), -1);
  }
  {
    auto Visitor = makeVisitor([](auto) { return -1; },
                               [](int I) { return I + 100; },
                               [](float F) { return F * 2; });
    EXPECT_EQ(Visitor(24), 124);
    EXPECT_EQ(Visitor(2.f), 4.f);
    EXPECT_EQ(Visitor(2.), -1);
    EXPECT_EQ(Visitor(Visitor), -1);
  }
}

template <bool Moveable, bool Copyable>
struct Functor : Counted<Moveable, Copyable> {
  using Counted<Moveable, Copyable>::Counted;
  void operator()() {}
};

TEST(STLExtrasTest, MakeVisitorLifetimeSemanticsPRValue) {
  int Copies = 0;
  int Moves = 0;
  int Destructors = 0;
  {
    auto V = makeVisitor(Functor<true, false>(Copies, Moves, Destructors));
    (void)V;
    EXPECT_EQ(0, Copies);
    EXPECT_EQ(1, Moves);
    EXPECT_EQ(1, Destructors);
  }
  EXPECT_EQ(0, Copies);
  EXPECT_EQ(1, Moves);
  EXPECT_EQ(2, Destructors);
}

TEST(STLExtrasTest, MakeVisitorLifetimeSemanticsRValue) {
  int Copies = 0;
  int Moves = 0;
  int Destructors = 0;
  {
    Functor<true, false> F(Copies, Moves, Destructors);
    {
      auto V = makeVisitor(std::move(F));
      (void)V;
      EXPECT_EQ(0, Copies);
      EXPECT_EQ(1, Moves);
      EXPECT_EQ(0, Destructors);
    }
    EXPECT_EQ(0, Copies);
    EXPECT_EQ(1, Moves);
    EXPECT_EQ(1, Destructors);
  }
  EXPECT_EQ(0, Copies);
  EXPECT_EQ(1, Moves);
  EXPECT_EQ(2, Destructors);
}

TEST(STLExtrasTest, MakeVisitorLifetimeSemanticsLValue) {
  int Copies = 0;
  int Moves = 0;
  int Destructors = 0;
  {
    Functor<true, true> F(Copies, Moves, Destructors);
    {
      auto V = makeVisitor(F);
      (void)V;
      EXPECT_EQ(1, Copies);
      EXPECT_EQ(0, Moves);
      EXPECT_EQ(0, Destructors);
    }
    EXPECT_EQ(1, Copies);
    EXPECT_EQ(0, Moves);
    EXPECT_EQ(1, Destructors);
  }
  EXPECT_EQ(1, Copies);
  EXPECT_EQ(0, Moves);
  EXPECT_EQ(2, Destructors);
}

TEST(STLExtrasTest, AllOfZip) {
  std::vector<int> v1 = {0, 4, 2, 1};
  std::vector<int> v2 = {1, 4, 3, 6};
  EXPECT_TRUE(all_of_zip(v1, v2, [](int v1, int v2) { return v1 <= v2; }));
  EXPECT_FALSE(all_of_zip(v1, v2, [](int L, int R) { return L < R; }));

  // Triple vectors
  std::vector<int> v3 = {1, 6, 5, 7};
  EXPECT_EQ(true, all_of_zip(v1, v2, v3, [](int a, int b, int c) {
              return a <= b && b <= c;
            }));
  EXPECT_EQ(false, all_of_zip(v1, v2, v3, [](int a, int b, int c) {
              return a < b && b < c;
            }));

  // Shorter vector should fail even with an always-true predicate.
  std::vector<int> v_short = {1, 4};
  EXPECT_EQ(false, all_of_zip(v1, v_short, [](int, int) { return true; }));
  EXPECT_EQ(false,
            all_of_zip(v1, v2, v_short, [](int, int, int) { return true; }));
}

TEST(STLExtrasTest, TypesAreDistinct) {
  EXPECT_TRUE((llvm::TypesAreDistinct<>::value));
  EXPECT_TRUE((llvm::TypesAreDistinct<int>::value));
  EXPECT_FALSE((llvm::TypesAreDistinct<int, int>::value));
  EXPECT_TRUE((llvm::TypesAreDistinct<int, float>::value));
  EXPECT_FALSE((llvm::TypesAreDistinct<int, float, int>::value));
  EXPECT_TRUE((llvm::TypesAreDistinct<int, float, double>::value));
  EXPECT_FALSE((llvm::TypesAreDistinct<int, float, double, float>::value));
  EXPECT_TRUE((llvm::TypesAreDistinct<int, int *>::value));
  EXPECT_TRUE((llvm::TypesAreDistinct<int, int &>::value));
  EXPECT_TRUE((llvm::TypesAreDistinct<int, int &&>::value));
  EXPECT_TRUE((llvm::TypesAreDistinct<int, const int>::value));
}

TEST(STLExtrasTest, FirstIndexOfType) {
  EXPECT_EQ((llvm::FirstIndexOfType<int, int>::value), 0u);
  EXPECT_EQ((llvm::FirstIndexOfType<int, int, int>::value), 0u);
  EXPECT_EQ((llvm::FirstIndexOfType<int, float, int>::value), 1u);
  EXPECT_EQ((llvm::FirstIndexOfType<int const *, float, int, int const *,
                                    const int>::value),
            2u);
}

TEST(STLExtrasTest, TypeAtIndex) {
  EXPECT_TRUE((std::is_same<int, llvm::TypeAtIndex<0, int>>::value));
  EXPECT_TRUE((std::is_same<int, llvm::TypeAtIndex<0, int, float>>::value));
  EXPECT_TRUE((std::is_same<float, llvm::TypeAtIndex<1, int, float>>::value));
  EXPECT_TRUE(
      (std::is_same<float, llvm::TypeAtIndex<1, int, float, double>>::value));
  EXPECT_TRUE(
      (std::is_same<float, llvm::TypeAtIndex<1, int, float, double>>::value));
  EXPECT_TRUE(
      (std::is_same<double, llvm::TypeAtIndex<2, int, float, double>>::value));
}

} // namespace
