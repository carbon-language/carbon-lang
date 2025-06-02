// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/ids.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>
#include <tuple>

namespace Carbon::SemIR {
namespace {

using ::testing::Eq;

TEST(IdsTest, LocIdValues) {
  // This testing should match the ranges documented on LocId.
  EXPECT_THAT(static_cast<LocId>(Parse::NodeId::None).index, Eq(-1));

  EXPECT_THAT(static_cast<LocId>(InstId(0)).index, Eq(0));
  EXPECT_THAT(
      static_cast<LocId>(InstId(std::numeric_limits<int32_t>::max())).index,
      Eq(std::numeric_limits<int32_t>::max()));

  EXPECT_THAT(static_cast<LocId>(Parse::NodeId(0)).index, Eq(-2));
  EXPECT_THAT(static_cast<LocId>(Parse::NodeId(Parse::NodeId::Max - 1)).index,
              Eq(-2 - (1 << 24) + 1));

  EXPECT_THAT(static_cast<LocId>(ImportIRInstId(0)).index, Eq(-2 - (1 << 24)));
  EXPECT_THAT(static_cast<LocId>(ImportIRInstId(ImportIRInstId::Max - 1)).index,
              Eq(-(1 << 30) + 1));
}

// A standard parameterized test for (is_desugared, index).
class IdsTestWithParam
    : public testing::TestWithParam<std::tuple<bool, int32_t>> {
 public:
  explicit IdsTestWithParam() {
    llvm::errs() << "is_desugared=" << is_desugared() << ", index=" << index()
                 << "\n";
  }

  // Returns IdT with its matching LocId form. Sets flags based on test
  // parameters.
  template <typename IdT>
  auto BuildIdAndLocId() -> std::pair<IdT, LocId> {
    IdT id(index());
    if (is_desugared()) {
      return {id, LocId(id).AsDesugared()};
    } else {
      return {id, LocId(id)};
    }
  }

  auto is_desugared() -> bool { return std::get<0>(GetParam()); }
  auto index() -> int32_t { return std::get<1>(GetParam()); }
};

// Returns a test case generator for edge-case values.
static auto GetValueRange(int32_t max) -> auto {
  return testing::Values(0, 1, max - 2, max - 1);
}

// Returns a test case generator for `IdsTestWithParam` uses.
static auto CombineWithFlags(auto value_range) -> auto {
  return testing::Combine(testing::Bool(), value_range);
}

class LocIdAsNoneTestWithParam : public IdsTestWithParam {};

INSTANTIATE_TEST_SUITE_P(
    LocIdAsNoneTest, LocIdAsNoneTestWithParam,
    CombineWithFlags(testing::Values(Parse::NodeId::NoneIndex)));

TEST_P(LocIdAsNoneTestWithParam, Test) {
  auto [_, loc_id] = BuildIdAndLocId<Parse::NodeId>();
  EXPECT_FALSE(loc_id.has_value());
  EXPECT_THAT(loc_id.kind(), Eq(LocId::Kind::None));
  EXPECT_FALSE(loc_id.is_desugared());
  EXPECT_THAT(loc_id.import_ir_inst_id(), Eq(ImportIRInstId::None));
  EXPECT_THAT(loc_id.inst_id(), Eq(InstId::None));
  EXPECT_THAT(loc_id.node_id(),
              // The actual type is NoneNodeId, so cast to NodeId.
              Eq<Parse::NodeId>(Parse::NodeId::None));
}

class LocIdAsImportIRInstIdTest : public IdsTestWithParam {};

INSTANTIATE_TEST_SUITE_P(Test, LocIdAsImportIRInstIdTest,
                         CombineWithFlags(GetValueRange(ImportIRInstId::Max)));

TEST_P(LocIdAsImportIRInstIdTest, Test) {
  auto [import_ir_inst_id, loc_id] = BuildIdAndLocId<ImportIRInstId>();
  EXPECT_TRUE(loc_id.has_value());
  ASSERT_THAT(loc_id.kind(), Eq(LocId::Kind::ImportIRInstId));
  EXPECT_THAT(loc_id.import_ir_inst_id(), import_ir_inst_id);
  EXPECT_FALSE(loc_id.is_desugared());
}

class LocIdAsInstIdTest : public IdsTestWithParam {};

INSTANTIATE_TEST_SUITE_P(
    Test, LocIdAsInstIdTest,
    testing::Combine(testing::Values(false),
                     GetValueRange(std::numeric_limits<int32_t>::max())));

TEST_P(LocIdAsInstIdTest, Test) {
  auto [inst_id, loc_id] = BuildIdAndLocId<InstId>();
  EXPECT_TRUE(loc_id.has_value());
  ASSERT_THAT(loc_id.kind(), Eq(LocId::Kind::InstId));
  EXPECT_THAT(loc_id.inst_id(), inst_id);
  // Note that `is_desugared` is invalid to use with `InstId`.
}

class LocIdAsNodeIdTest : public IdsTestWithParam {};

INSTANTIATE_TEST_SUITE_P(Test, LocIdAsNodeIdTest,
                         CombineWithFlags(GetValueRange(Parse::NodeId::Max)));

TEST_P(LocIdAsNodeIdTest, Test) {
  auto [node_id, loc_id] = BuildIdAndLocId<Parse::NodeId>();
  EXPECT_TRUE(loc_id.has_value());
  ASSERT_THAT(loc_id.kind(), Eq(LocId::Kind::NodeId));
  EXPECT_THAT(loc_id.node_id(), node_id);
  EXPECT_THAT(loc_id.is_desugared(), Eq(is_desugared()));
}

}  // namespace
}  // namespace Carbon::SemIR
