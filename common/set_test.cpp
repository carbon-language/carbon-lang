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
auto ExpectSetElementsAre(SetT&& s, MatcherRangeT element_matchers) -> void {
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
auto ExpectSetElementsAre(SetT&& s,
                          std::initializer_list<MatcherT> element_matchers)
    -> void {
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

  // Copy-assign updates.
  other_s1 = s;
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 32)));

  // Self-assign is a no-op.
  other_s1 = const_cast<const SetT&>(other_s1);
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 32)));

  // But mutating original still doesn't change copies.
  for (int i : llvm::seq(32, 48)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 32)));
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

  // Move back over a moved-from.
  s = std::move(other_s1);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(1, 32)));

  // Copy over moved-from state also works.
  other_s1 = s;
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 32)));

  // Now add still more elements.
  for (int i : llvm::seq(32, 48)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(other_s1.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 48)));

  // Move-assign over the copy looks like the moved-from table not the copy.
  other_s1 = std::move(s);
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 32)));

  // Self-swap (which does a self-move) works and is a no-op.
  std::swap(other_s1, other_s1);
  ExpectSetElementsAre(other_s1, MakeElements(llvm::seq(1, 32)));

  // Test copying of a moved-from table over a valid table and self-move-assign.
  // The former is required to be valid, and the latter is in at least the case
  // of self-move-assign-when-moved-from, but the result can be in any state so
  // just do them and ensure we don't crash.
  SetT other_s2 = other_s1;
  // NOLINTNEXTLINE(bugprone-use-after-move): Testing required use-after-move.
  other_s2 = s;
  other_s1 = std::move(other_s1);
  s = std::move(s);
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

TYPED_TEST(SetTest, GrowToAllocSize) {
  using SetT = TypeParam;

  SetT s;
  // Grow when empty. May be a no-op for some small sizes.
  s.GrowToAllocSize(32);

  // Add some elements that will need to be propagated through subsequent
  // growths. Also delete some.
  ssize_t storage_bytes = s.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(1, 24)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 8)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Erase(i));
  }
  // No further growth triggered.
  EXPECT_EQ(storage_bytes, s.ComputeMetrics().storage_bytes);

  // No-op.
  s.GrowToAllocSize(16);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(8, 24)));
  // No further growth triggered.
  EXPECT_EQ(storage_bytes, s.ComputeMetrics().storage_bytes);

  // Get a few doubling based growths, and at least one beyond the largest small
  // size.
  s.GrowToAllocSize(64);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(8, 24)));
  s.GrowToAllocSize(128);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(8, 24)));
  s.GrowToAllocSize(256);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(8, 24)));
  // Update the storage bytes after growth.
  EXPECT_LT(storage_bytes, s.ComputeMetrics().storage_bytes);
  storage_bytes = s.ComputeMetrics().storage_bytes;

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
  // No growth from insertions.
  EXPECT_EQ(storage_bytes, s.ComputeMetrics().storage_bytes);

  s.GrowToAllocSize(1024);
  ExpectSetElementsAre(s, MakeElements(llvm::seq(16, 48)));
  // Storage should have grown.
  EXPECT_LT(storage_bytes, s.ComputeMetrics().storage_bytes);
}

TYPED_TEST(SetTest, GrowForInsert) {
  using SetT = TypeParam;

  SetT s;
  s.GrowForInsertCount(42);
  ssize_t storage_bytes = s.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(1, 42)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(s, MakeElements(llvm::seq(1, 42)));
  EXPECT_EQ(storage_bytes, s.ComputeMetrics().storage_bytes);

  // Erase many elements and grow again for another insert.
  for (int i : llvm::seq(1, 32)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Erase(i));
  }
  s.GrowForInsertCount(42);
  storage_bytes = s.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(42, 84)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(s, MakeElements(llvm::seq(32, 84)));
  EXPECT_EQ(storage_bytes, s.ComputeMetrics().storage_bytes);

  // Erase all the elements, then grow for a much larger insertion and insert
  // again.
  for (int i : llvm::seq(32, 84)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Erase(i));
  }
  s.GrowForInsertCount(321);
  storage_bytes = s.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(128, 321 + 128)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(s.Insert(i).is_inserted());
  }
  ExpectSetElementsAre(s, MakeElements(llvm::seq(128, 321 + 128)));
  EXPECT_EQ(storage_bytes, s.ComputeMetrics().storage_bytes);
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
  EXPECT_TRUE(s.Insert(
                   TestData(200), [] { return 2; }, key_context)
                  .is_inserted());
  EXPECT_TRUE(s.Contains(2, key_context));
  EXPECT_TRUE(s.Contains(TestData(200), key_context));

  // Verify all the elements.
  ExpectSetElementsAre(s, {1, 2});

  // Fill up a bunch to ensure we trigger growth a few times.
  for (int i : llvm::seq(3, 512)) {
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
  EXPECT_FALSE(s.Contains(TestData(0), key_context));
  EXPECT_FALSE(s.Contains(TestData(51200), key_context));

  // Verify all the elements.
  ExpectSetElementsAre(s, MakeElements(llvm::seq(1, 512)));
}

}  // namespace
}  // namespace Carbon
