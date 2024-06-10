// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/set.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <type_traits>
#include <vector>

#include "common/raw_hashtable_test_helpers.h"

namespace Carbon {
namespace {

using RawHashtable::IndexKeyContext;
using RawHashtable::TestData;
using ::testing::UnorderedElementsAreArray;

template <typename SetT, typename MatcherRangeT>
void ExpectSetElementsAre(SetT&& s, MatcherRangeT element_matchers) {
  // Collect the elements into a container.
  using KeyT = typename std::remove_reference<SetT>::type::KeyT;
  std::vector<KeyT> entries;
  s.ForEach([&entries](KeyT& k) { entries.push_back(k); });

  // Use the GoogleMock unordered container matcher to validate and show errors
  // on wrong elements.
  EXPECT_THAT(entries, UnorderedElementsAreArray(element_matchers));
}

// Allow directly using an initializer list.
template <typename SetT, typename MatcherT>
void ExpectSetElementsAre(SetT&& s,
                          std::initializer_list<MatcherT> element_matchers) {
  std::vector<MatcherT> element_matchers_storage = element_matchers;
  ExpectSetElementsAre(s, element_matchers_storage);
}

template <typename RangeT, typename... RangeTs>
auto MakeElements(RangeT&& range, RangeTs&&... ranges) {
  std::vector<typename RangeT::value_type> elements;
  auto add_range = [&elements](RangeT&& r) {
    for (const auto&& e : r) {
      elements.push_back(e);
    }
  };
  add_range(std::forward<RangeT>(range));
  (add_range(std::forward<RangeT>(ranges)), ...);

  return elements;
}

template <typename SetT>
class SetTest : public ::testing::Test {};

using Types = ::testing::Types<Set<int>, Set<int, 16>, Set<int, 128>,
                               Set<TestData>, Set<TestData, 16>>;
TYPED_TEST_SUITE(SetTest, Types);

TYPED_TEST(SetTest, Basic) {
  using SetT = TypeParam;
  SetT s;

  EXPECT_FALSE(s.Contains(42));
  EXPECT_TRUE(s.Insert(1).is_inserted());
  EXPECT_TRUE(s.Contains(1));
  auto result = s.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  auto i_result = s.Insert(1);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_TRUE(s.Contains(1));

  // Verify all the elements.
  ExpectSetElementsAre(s, {1});

  // Fill up a bunch to ensure we trigger growth a few times.
  for (int i : llvm::seq(2, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
  }
  EXPECT_FALSE(s.Contains(513));

  // Verify all the elements.
  ExpectSetElementsAre(s, MakeElements(llvm::seq(1, 512)));
}

TYPED_TEST(SetTest, FactoryAPI) {
  using SetT = TypeParam;
  SetT s;
  EXPECT_TRUE(s.Insert(1, [](int k, void* key_storage) {
                 return new (key_storage) int(k);
               }).is_inserted());
  ASSERT_TRUE(s.Contains(1));
  // Reinsertion doesn't invoke the callback.
  EXPECT_FALSE(s.Insert(1, [](int, void*) -> int* {
                  llvm_unreachable("Should never be called!");
                }).is_inserted());
}

TYPED_TEST(SetTest, Copy) {
  using SetT = TypeParam;

  SetT s;
  // Make sure we exceed the small size for some of the set types, but not all
  // of them, so we cover all the combinations of copying between small and
  // large.
  for (int i : llvm::seq(1, 24)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }

  SetT other_s1 = s;
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 24)));

  // Add some more elements to the original.
  for (int i : llvm::seq(24, 32)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }

  // The first copy doesn't change.
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 24)));

  // A new copy does.
  SetT other_s2 = s;
  ExpectSetElementsAre(other_s2, MakeElements(llvm::seq(1, 32)));
}

TYPED_TEST(SetTest, Move) {
  using SetT = TypeParam;

  SetT s;
  // Make sure we exceed the small size for some of the set types, but not all
  // of them, so we cover all the combinations of copying between small and
  // large.
  for (int i : llvm::seq(1, 24)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }

  SetT other_s1 = std::move(s);
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 24)));

  // Add some more elements.
  for (int i : llvm::seq(24, 32)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(other_s1.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 32)));
}

TYPED_TEST(SetTest, Conversions) {
  using SetT = TypeParam;
  using KeyT = SetT::KeyT;
  SetT s;
  ASSERT_TRUE(s.Insert(1).is_inserted());
  ASSERT_TRUE(s.Insert(2).is_inserted());
  ASSERT_TRUE(s.Insert(3).is_inserted());
  ASSERT_TRUE(s.Insert(4).is_inserted());

  SetView<KeyT> sv = s;
  SetView<const KeyT> csv = sv;
  SetView<const KeyT> csv2 = s;
  EXPECT_TRUE(sv.Contains(1));
  EXPECT_TRUE(csv.Contains(2));
  EXPECT_TRUE(csv2.Contains(3));
}

TYPED_TEST(SetTest, Grow) {
  using SetT = TypeParam;

  SetT s;
  // Grow when empty. May be a no-op for some small sizes.
  s.Grow(32);

  // Add some elements that will need to be propagated through subsequent
  // growths. Also delete some.
  for (int i : llvm::seq(1, 24)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 8)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Erase(i));
  }

  // No-op.
  s.Grow(16);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(8, 24)));

  // Get a couple of doubling based growths.
  s.Grow(64);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(8, 24)));
  s.Grow(128);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(8, 24)));

  // Add some more, but not enough to trigger further growth, and then grow by
  // several more multiples of two to test handling large growth.
  for (int i : llvm::seq(24, 48)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(8, 16)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Erase(i));
  }
  s.Grow(1024);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(16, 48)));
}

TYPED_TEST(SetTest, GrowForInsert) {
  using SetT = TypeParam;

  SetT s;
  s.GrowForInsertCount(42);
  for (int i : llvm::seq(1, 42)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(s, MakeElements(llvm::seq(1, 42)));

  // Erase many elements and grow again for another insert.
  for (int i : llvm::seq(1, 32)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Erase(i));
  }
  s.GrowForInsertCount(42);
  for (int i : llvm::seq(42, 84)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(s, MakeElements(llvm::seq(32, 84)));

  // Erase all the elements, then grow for a much larger insertion and insert
  // again.
  for (int i : llvm::seq(32, 84)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Erase(i));
  }
  s.GrowForInsertCount(1717);
  for (int i : llvm::seq(128, 1717 + 128)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(s, MakeElements(llvm::seq(128, 1717 + 128)));
}

TEST(SetContextTest, Basic) {
  llvm::SmallVector<TestData> keys;
  for (int i : llvm::seq(0, 513)) {
    keys.push_back(i * 100);
  }
  IndexKeyContext<TestData> key_context(keys);
  Set<ssize_t, 0, IndexKeyContext<TestData>> s;

  EXPECT_FALSE(s.Contains(42, key_context));
  EXPECT_TRUE(s.Insert(1, key_context).is_inserted());
  EXPECT_TRUE(s.Contains(1, key_context));
  auto result = s.Lookup(TestData(100), key_context);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  auto i_result = s.Insert(1, IndexKeyContext<TestData>(keys));
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_TRUE(s.Contains(1, key_context));

  // Verify all the elements.
  ExpectSetElementsAre(s, {1});

  // Fill up a bunch to ensure we trigger growth a few times.
  for (int i : llvm::seq(2, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Insert(i, key_context).is_inserted());
  }
  for (int i : llvm::seq(1, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i, key_context));
    EXPECT_FALSE(s.Insert(i, key_context).is_inserted());
  }
  EXPECT_FALSE(s.Contains(0, key_context));
  EXPECT_FALSE(s.Contains(512, key_context));

  // Verify all the elements.
  ExpectSetElementsAre(s, MakeElements(llvm::seq(1, 512)));
}

}  // namespace
}  // namespace Carbon
