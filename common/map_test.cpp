// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/map.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/raw_hashtable_test_helpers.h"

namespace Carbon::Testing {
namespace {

using RawHashtable::FixedHashKeyContext;
using RawHashtable::IndexKeyContext;
using RawHashtable::TestData;
using RawHashtable::TestKeyContext;
using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

template <typename MapT, typename MatcherRangeT>
auto ExpectMapElementsAre(MapT&& m, MatcherRangeT element_matchers) -> void {
  // Now collect the elements into a container.
  using KeyT = typename std::remove_reference<MapT>::type::KeyT;
  using ValueT = typename std::remove_reference<MapT>::type::ValueT;
  std::vector<std::pair<KeyT, ValueT>> map_entries;
  m.ForEach([&map_entries](KeyT& k, ValueT& v) {
    map_entries.push_back({k, v});
  });

  // Use the GoogleMock unordered container matcher to validate and show errors
  // on wrong elements.
  EXPECT_THAT(map_entries, UnorderedElementsAreArray(element_matchers));
}

// Allow directly using an initializer list.
template <typename MapT, typename MatcherT>
auto ExpectMapElementsAre(MapT&& m,
                          std::initializer_list<MatcherT> element_matchers)
    -> void {
  std::vector<MatcherT> element_matchers_storage = element_matchers;
  ExpectMapElementsAre(m, element_matchers_storage);
}

template <typename ValueCB, typename RangeT, typename... RangeTs>
auto MakeKeyValues(ValueCB value_cb, RangeT&& range, RangeTs&&... ranges)
    -> auto {
  using KeyT = typename RangeT::value_type;
  using ValueT = decltype(value_cb(std::declval<KeyT>()));
  std::vector<std::pair<KeyT, ValueT>> elements;
  auto add_range = [&](RangeT&& r) {
    for (const auto&& e : r) {
      elements.push_back({e, value_cb(e)});
    }
  };
  add_range(std::forward<RangeT>(range));
  (add_range(std::forward<RangeT>(ranges)), ...);

  return elements;
}

template <typename MapT>
class MapTest : public ::testing::Test {};

using Types = ::testing::Types<
    Map<int, int>, Map<int, int, 16>, Map<int, int, 64>,
    Map<int, int, 0, TestKeyContext>, Map<int, int, 16, TestKeyContext>,
    Map<int, int, 64, TestKeyContext>, Map<TestData, TestData>,
    Map<TestData, TestData, 16>, Map<TestData, TestData, 0, TestKeyContext>,
    Map<TestData, TestData, 16, TestKeyContext>>;
TYPED_TEST_SUITE(MapTest, Types);

TYPED_TEST(MapTest, Basic) {
  TypeParam m;

  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);
  EXPECT_TRUE(m.Insert(1, 100).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  auto result = m.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  EXPECT_EQ(100, *m[1]);
  // Reinsertion doesn't change the value.
  auto i_result = m.Insert(1, 101);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(100, i_result.value());
  EXPECT_EQ(100, *m[1]);
  // Update does change the value.
  i_result = m.Update(1, 101);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(101, i_result.value());
  EXPECT_EQ(101, *m[1]);

  // Verify all the elements.
  ExpectMapElementsAre(m, {Pair(1, 101)});

  // Fill up a bunch to ensure we trigger growth a few times.
  for (int i : llvm::seq(2, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + static_cast<int>(k == 1); },
                       llvm::seq(1, 512)));
  for (int i : llvm::seq(1, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + static_cast<int>(i == 1), *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
  }
  EXPECT_FALSE(m.Contains(513));

  // Verify all the elements.
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(1, 512)));
}

TYPED_TEST(MapTest, FactoryAPI) {
  TypeParam m;
  EXPECT_TRUE(m.Insert(1, [] { return 100; }).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  EXPECT_EQ(100, *m[1]);
  // Reinsertion doesn't invoke the callback.
  EXPECT_FALSE(m.Insert(1, []() -> int {
                  llvm_unreachable("Should never be called!");
                }).is_inserted());
  // Update does invoke the callback.
  auto i_result = m.Update(1, [] { return 101; });
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(101, i_result.value());
  EXPECT_EQ(101, *m[1]);
}

TYPED_TEST(MapTest, Copy) {
  using MapT = TypeParam;

  MapT m;
  // Make sure we exceed the small size for some of the map types, but not all
  // of them, so we cover all the combinations of copying between small and
  // large.
  for (int i : llvm::seq(1, 24)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }

  MapT other_m1 = m;
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 24)));

  // Add some more elements to the original.
  for (int i : llvm::seq(24, 32)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }

  // The first copy doesn't change.
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 24)));

  // A new copy does.
  MapT other_m2 = m;
  ExpectMapElementsAre(
      other_m2, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // Copy-assign updates.
  other_m1 = m;
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // Self-assign is a no-op.
  other_m1 = const_cast<const MapT&>(other_m1);
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // But mutating original still doesn't change copies.
  for (int i : llvm::seq(32, 48)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));
  ExpectMapElementsAre(
      other_m2, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));
}

TYPED_TEST(MapTest, Move) {
  using MapT = TypeParam;

  MapT m;
  // Make sure we exceed the small size for some of the map types, but not all
  // of them, so we cover all the combinations of moving between small and
  // large.
  for (int i : llvm::seq(1, 24)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }

  MapT other_m1 = std::move(m);
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 24)));

  // Add some more elements.
  for (int i : llvm::seq(24, 32)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(other_m1.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // Move back over a moved-from.
  m = std::move(other_m1);
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // Copy over moved-from state also works.
  other_m1 = m;
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // Now add still more elements.
  for (int i : llvm::seq(32, 48)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(other_m1.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 48)));

  // And move-assign over the copy looks like the moved-from table not the copy.
  other_m1 = std::move(m);
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // Self-swap (which does a self-move) works and is a no-op.
  std::swap(other_m1, other_m1);
  ExpectMapElementsAre(
      other_m1, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 32)));

  // Test copying of a moved-from table over a valid table and self-move-assign.
  // The former is required to be valid, and the latter is in at least the case
  // of self-move-assign-when-moved-from, but the result can be in any state so
  // just do them and ensure we don't crash.
  MapT other_m2 = other_m1;
  // NOLINTNEXTLINE(bugprone-use-after-move): Testing required use-after-move.
  other_m2 = m;
  other_m1 = std::move(other_m1);
  m = std::move(m);
}

TYPED_TEST(MapTest, Conversions) {
  using MapT = TypeParam;
  using KeyT = MapT::KeyT;
  using ValueT = MapT::ValueT;
  using KeyContextT = MapT::KeyContextT;

  MapT m;

  ASSERT_TRUE(m.Insert(1, 101).is_inserted());
  ASSERT_TRUE(m.Insert(2, 102).is_inserted());
  ASSERT_TRUE(m.Insert(3, 103).is_inserted());
  ASSERT_TRUE(m.Insert(4, 104).is_inserted());

  MapView<KeyT, ValueT, KeyContextT> mv = m;
  MapView<const KeyT, ValueT, KeyContextT> cmv = m;
  MapView<KeyT, const ValueT, KeyContextT> cmv2 = m;
  MapView<const KeyT, const ValueT, KeyContextT> cmv3 = m;
  EXPECT_TRUE(mv.Contains(1));
  EXPECT_EQ(101, *mv[1]);
  EXPECT_TRUE(cmv.Contains(2));
  EXPECT_EQ(102, *cmv[2]);
  EXPECT_TRUE(cmv2.Contains(3));
  EXPECT_EQ(103, *cmv2[3]);
  EXPECT_TRUE(cmv3.Contains(4));
  EXPECT_EQ(104, *cmv3[4]);
}

TYPED_TEST(MapTest, GrowToAllocSize) {
  using MapT = TypeParam;

  MapT m;
  // Grow when empty. May be a no-op for some small sizes.
  m.GrowToAllocSize(32);

  // Add some elements that will need to be propagated through subsequent
  // growths. Also delete some.
  ssize_t storage_bytes = m.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(1, 24)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(1, 8)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Erase(i));
  }
  // No further growth triggered.
  EXPECT_EQ(storage_bytes, m.ComputeMetrics().storage_bytes);

  // No-op.
  m.GrowToAllocSize(16);
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(8, 24)));
  // No further growth triggered.
  EXPECT_EQ(storage_bytes, m.ComputeMetrics().storage_bytes);

  // Get a few doubling based growths, and at least one beyond the largest small
  // size.
  m.GrowToAllocSize(64);
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(8, 24)));
  m.GrowToAllocSize(128);
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(8, 24)));
  // Update the storage bytes after growth.
  EXPECT_LT(storage_bytes, m.ComputeMetrics().storage_bytes);
  storage_bytes = m.ComputeMetrics().storage_bytes;

  // Add some more, but not enough to trigger further growth, and then grow by
  // several more multiples of two to test handling large growth.
  for (int i : llvm::seq(24, 48)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(8, 16)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Erase(i));
  }
  // No growth from insertions.
  EXPECT_EQ(storage_bytes, m.ComputeMetrics().storage_bytes);

  m.GrowToAllocSize(1024);
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(16, 48)));
  // Storage should have grown.
  EXPECT_LT(storage_bytes, m.ComputeMetrics().storage_bytes);
}

TYPED_TEST(MapTest, GrowForInsert) {
  using MapT = TypeParam;

  MapT m;
  m.GrowForInsertCount(42);
  ssize_t storage_bytes = m.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(1, 42)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 42)));
  EXPECT_EQ(storage_bytes, m.ComputeMetrics().storage_bytes);

  // Erase many elements and grow again for another insert.
  for (int i : llvm::seq(1, 32)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Erase(i));
  }
  m.GrowForInsertCount(42);
  storage_bytes = m.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(42, 84)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(32, 84)));
  EXPECT_EQ(storage_bytes, m.ComputeMetrics().storage_bytes);

  // Erase all the elements, then grow for a much larger insertion and insert
  // again.
  for (int i : llvm::seq(32, 84)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Erase(i));
  }
  m.GrowForInsertCount(321);
  storage_bytes = m.ComputeMetrics().storage_bytes;
  for (int i : llvm::seq(128, 321 + 128)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(m, MakeKeyValues([](int k) { return k * 100; },
                                        llvm::seq(128, 321 + 128)));
  EXPECT_EQ(storage_bytes, m.ComputeMetrics().storage_bytes);
}

// This test is largely exercising the underlying `RawHashtable` implementation
// with complex growth, erasure, and re-growth.
TYPED_TEST(MapTest, ComplexOpSequence) {
  // Use a small size as well to cover more growth scenarios.
  TypeParam m;

  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);
  EXPECT_TRUE(m.Insert(1, 100).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  auto result = m.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  EXPECT_EQ(100, *m[1]);
  // Reinsertion doesn't change the value.
  auto i_result = m.Insert(1, 101);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(100, i_result.value());
  EXPECT_EQ(100, *m[1]);
  // Update does change the value.
  i_result = m.Update(1, 101);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(101, i_result.value());
  EXPECT_EQ(101, *m[1]);

  // Verify all the elements.
  ExpectMapElementsAre(m, {Pair(1, 101)});

  // Fill up the small buffer but don't overflow it.
  for (int i : llvm::seq(2, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(1, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + static_cast<int>(i == 1), *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + static_cast<int>(i == 1), *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
  }
  EXPECT_FALSE(m.Contains(5));

  // Verify all the elements.
  ExpectMapElementsAre(
      m, {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});

  // Erase some entries from the small buffer.
  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Erase(2));
  EXPECT_EQ(101, *m[1]);
  EXPECT_EQ(nullptr, m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(401, *m[4]);
  EXPECT_TRUE(m.Erase(1));
  EXPECT_EQ(nullptr, m[1]);
  EXPECT_EQ(nullptr, m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(401, *m[4]);
  EXPECT_TRUE(m.Erase(4));
  EXPECT_EQ(nullptr, m[1]);
  EXPECT_EQ(nullptr, m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(nullptr, m[4]);
  // Fill them back in, but with a different order and going back to the
  // original value.
  EXPECT_TRUE(m.Insert(1, 100).is_inserted());
  EXPECT_TRUE(m.Insert(2, 200).is_inserted());
  EXPECT_TRUE(m.Insert(4, 400).is_inserted());
  EXPECT_EQ(100, *m[1]);
  EXPECT_EQ(200, *m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(400, *m[4]);
  // Then update their values to match.
  EXPECT_FALSE(m.Update(1, 101).is_inserted());
  EXPECT_FALSE(m.Update(2, 201).is_inserted());
  EXPECT_FALSE(m.Update(4, 401).is_inserted());

  // Now fill up the first metadata group.
  for (int i : llvm::seq(5, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(1, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + static_cast<int>(i < 5), *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + static_cast<int>(i < 5), *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 2, *m[i]);
  }
  EXPECT_FALSE(m.Contains(42));

  // Verify all the elements by walking the entire map.
  ExpectMapElementsAre(
      m, {Pair(1, 102), Pair(2, 202), Pair(3, 302), Pair(4, 402), Pair(5, 502),
          Pair(6, 602), Pair(7, 702), Pair(8, 802), Pair(9, 902),
          Pair(10, 1002), Pair(11, 1102), Pair(12, 1202), Pair(13, 1302)});

  // Now fill up several more groups.
  for (int i : llvm::seq(14, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(1, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + 2 * static_cast<int>(i < 14), *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 2 * static_cast<int>(i < 14), *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 3).is_inserted());
    EXPECT_EQ(i * 100 + 3, *m[i]);
  }
  EXPECT_FALSE(m.Contains(420));

  // Check walking the entire container.
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 3; }, llvm::seq(1, 100)));

  // Clear back to empty.
  m.Clear();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150)) {
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(50, 150)));

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(73));
  EXPECT_TRUE(m.Erase(73));
  EXPECT_FALSE(m.Contains(73));
  for (int i : llvm::seq(102, 136)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    if (i == 73 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 2, *m[i]);
  }
  EXPECT_TRUE(m.Insert(73, 73 * 100 + 3).is_inserted());
  EXPECT_EQ(73 * 100 + 3, *m[73]);

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 2 + (k == 73); },
                       llvm::seq(50, 102), llvm::seq(136, 150)));

  // Reset back to empty and small.
  m.Reset();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values, now triggering
  // growth too. Also, use update instead of insert.
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Update(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(75, 175)));

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(93));
  EXPECT_TRUE(m.Erase(93));
  EXPECT_FALSE(m.Contains(93));
  for (int i : llvm::seq(102, 136)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    if (i == 93 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 2, *m[i]);
  }
  EXPECT_TRUE(m.Insert(93, 93 * 100 + 3).is_inserted());
  EXPECT_EQ(93 * 100 + 3, *m[93]);

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 2 + (k == 93); },
                       llvm::seq(75, 102), llvm::seq(136, 175)));
}

template <typename MapT>
class MapCollisionTest : public ::testing::Test {};

using CollisionTypes = ::testing::Types<
    Map<int, int, 16,
        FixedHashKeyContext<7, /*FixIndexBits*/ true, /*FixTagBits*/ false, 0>>,
    Map<int, int, 16,
        FixedHashKeyContext<7, /*FixIndexBits*/ false, /*FixTagBits*/ true, 0>>,
    Map<int, int, 16,
        FixedHashKeyContext<7, /*FixIndexBits*/ true, /*FixTagBits*/ true, 0>>,
    Map<int, int, 16,
        FixedHashKeyContext<7, /*FixIndexBits*/ true, /*FixTagBits*/ true,
                            ~static_cast<uint64_t>(0)>>>;
TYPED_TEST_SUITE(MapCollisionTest, CollisionTypes);

TYPED_TEST(MapCollisionTest, Basic) {
  TypeParam m;

  // Fill the map through a couple of growth steps, verifying at each step. Note
  // that because this is a collision test, we synthesize actively harmful
  // hashes in terms of collisions and so this test is essentially quadratic. We
  // need to keep it relatively small.
  for (int i : llvm::seq(1, 256)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 256)));

  // Erase and re-fill from the back.
  for (int i : llvm::seq(192, 256)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Erase(i));
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100; }, llvm::seq(1, 192)));
  for (int i : llvm::seq(192, 256)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100 + 1).is_inserted());
  }
  ExpectMapElementsAre(m,
                       MakeKeyValues([](int k) { return k * 100 + (k >= 192); },
                                     llvm::seq(1, 256)));

  // Erase and re-fill from the front.
  for (int i : llvm::seq(1, 64)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Erase(i));
  }
  ExpectMapElementsAre(m,
                       MakeKeyValues([](int k) { return k * 100 + (k >= 192); },
                                     llvm::seq(64, 256)));
  for (int i : llvm::seq(1, 64)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100 + 1).is_inserted());
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + (k < 64) + (k >= 192); },
                       llvm::seq(1, 256)));

  // Erase and re-fill from the middle.
  for (int i : llvm::seq(64, 192)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Erase(i));
  }
  ExpectMapElementsAre(m, MakeKeyValues([](int k) { return k * 100 + 1; },
                                        llvm::seq(1, 64), llvm::seq(192, 256)));
  for (int i : llvm::seq(64, 192)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100 + 1).is_inserted());
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(1, 256)));

  // Erase and re-fill from both the back and front.
  for (auto s : {llvm::seq(192, 256), llvm::seq(1, 64)}) {
    for (int i : s) {
      SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
      EXPECT_TRUE(m.Erase(i));
    }
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(64, 192)));
  for (auto s : {llvm::seq(192, 256), llvm::seq(1, 64)}) {
    for (int i : s) {
      SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
      EXPECT_TRUE(m.Insert(i, i * 100 + 2).is_inserted());
    }
  }
  ExpectMapElementsAre(
      m,
      MakeKeyValues([](int k) { return k * 100 + 1 + (k < 64) + (k >= 192); },
                    llvm::seq(1, 256)));

  // And update the middle elements in place.
  for (int i : llvm::seq(64, 192)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted());
  }
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 2; }, llvm::seq(1, 256)));
}

TEST(MapContextTest, Basic) {
  llvm::SmallVector<TestData> keys;
  for (int i : llvm::seq(0, 513)) {
    keys.push_back(i * 100000);
  }
  IndexKeyContext<TestData> key_context(keys);
  Map<ssize_t, int, 0, IndexKeyContext<TestData>> m;

  EXPECT_FALSE(m.Contains(42, key_context));
  EXPECT_TRUE(m.Insert(1, 100, key_context).is_inserted());
  ASSERT_TRUE(m.Contains(1, key_context));
  auto result = m.Lookup(TestData(100000), key_context);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  // Reinsertion doesn't change the value. Also, double check a temporary
  // context.
  auto i_result = m.Insert(1, 101, IndexKeyContext<TestData>(keys));
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(100, i_result.value());
  // Update does change the value.
  i_result = m.Update(1, 101, key_context);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(101, i_result.value());

  // Verify all the elements.
  ExpectMapElementsAre(m, {Pair(1, 101)});

  // Fill up a bunch to ensure we trigger growth a few times.
  for (int i : llvm::seq(2, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100, key_context).is_inserted());
  }
  // Check all the elements, including using the context.
  for (int j : llvm::seq(1, 512)) {
    SCOPED_TRACE(llvm::formatv("Assert key: {0}", j).str());
    ASSERT_EQ(j * 100 + static_cast<int>(j == 1),
              m.Lookup(j, key_context).value());
    ASSERT_EQ(j * 100 + static_cast<int>(j == 1),
              m.Lookup(TestData(j * 100000), key_context).value());
  }
  for (int i : llvm::seq(1, 512)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_FALSE(m.Insert(i, i * 100 + 1, key_context).is_inserted());
    EXPECT_EQ(i * 100 + static_cast<int>(i == 1),
              m.Lookup(i, key_context).value());
    EXPECT_FALSE(m.Update(i, i * 100 + 1, key_context).is_inserted());
    EXPECT_EQ(i * 100 + 1, m.Lookup(i, key_context).value());
  }
  EXPECT_FALSE(m.Contains(0, key_context));
  EXPECT_FALSE(m.Contains(512, key_context));

  // Verify all the elements.
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(1, 512)));
}

}  // namespace
}  // namespace Carbon::Testing
