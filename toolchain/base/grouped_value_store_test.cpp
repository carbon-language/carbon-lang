// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/grouped_value_store.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utility>

#include "toolchain/base/index_base.h"

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

struct TestId : public IndexBase<TestId> {
  // Only used when a `TestId` is printed, for example by a failing matcher.
  [[maybe_unused]] static constexpr llvm::StringLiteral Label = "test";

  using IndexBase::IndexBase;
};

TEST(GroupedValueStore, Empty) {
  GroupedValueStore<TestId, int> store(0, [](auto /*add*/) {});

  EXPECT_EQ(store.size(), 0U);
  EXPECT_THAT(store.Get(TestId::None), IsEmpty());
}

TEST(GroupedValueStore, GroupsValuesById) {
  std::pair<int, char> values[] = {{2, 'a'}, {0, 'b'}, {2, 'c'}};
  GroupedValueStore<TestId, char> store(4, [&](auto add) {
    for (auto [index, value] : values) {
      add(TestId(index), value);
    }
  });

  EXPECT_EQ(store.size(), 3U);
  EXPECT_THAT(store.Get(TestId(0)), ElementsAre('b'));
  // A group with no values is empty, not missing.
  EXPECT_THAT(store.Get(TestId(1)), IsEmpty());
  // Values within a group are in the order they were enumerated.
  EXPECT_THAT(store.Get(TestId(2)), ElementsAre('a', 'c'));
  EXPECT_THAT(store.Get(TestId(3)), IsEmpty());
}

TEST(GroupedValueStore, IgnoresIdWithNoValue) {
  GroupedValueStore<TestId, char> store(2, [](auto add) {
    add(TestId::None, 'a');
    add(TestId(1), 'b');
  });

  EXPECT_EQ(store.size(), 1U);
  EXPECT_THAT(store.Get(TestId::None), IsEmpty());
  EXPECT_THAT(store.Get(TestId(1)), ElementsAre('b'));
}

TEST(GroupedValueStore, StoresIdValues) {
  // An ID type has no default constructor, so this exercises filling the value
  // array with `None` before overwriting it.
  GroupedValueStore<TestId, TestId> store(2, [](auto add) {
    add(TestId(0), TestId(7));
    add(TestId(0), TestId(8));
  });

  EXPECT_THAT(store.Get(TestId(0)), ElementsAre(TestId(7), TestId(8)));
  EXPECT_THAT(store.Get(TestId(1)), IsEmpty());
}

}  // namespace
}  // namespace Carbon::Testing
