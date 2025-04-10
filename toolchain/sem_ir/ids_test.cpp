// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/ids.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

namespace Carbon::SemIR {
namespace {

using ::testing::Eq;

TEST(IdsTest, LocIdIsNone) {
  LocId loc_id = Parse::NodeId::None;
  EXPECT_FALSE(loc_id.has_value());
  EXPECT_FALSE(loc_id.is_node_id());
  EXPECT_FALSE(loc_id.is_import_ir_inst_id());
  EXPECT_FALSE(loc_id.is_implicit());
  EXPECT_THAT(loc_id.node_id(),
              // The actual type is NoneNodeId, so cast to NodeId.
              Eq<Parse::NodeId>(Parse::NodeId::None));
  EXPECT_THAT(loc_id.import_ir_inst_id(), Eq(ImportIRInstId::None));
}

TEST(IdsTest, LocIdIsNodeId) {
  for (auto index : {0, 1, Parse::NodeId::Max - 2, Parse::NodeId::Max - 1}) {
    SCOPED_TRACE(llvm::formatv("Index: {0}", index));
    Parse::NodeId node_id(index);
    LocId loc_id = node_id;
    EXPECT_TRUE(loc_id.has_value());
    EXPECT_TRUE(loc_id.is_node_id());
    EXPECT_FALSE(loc_id.is_import_ir_inst_id());
    EXPECT_FALSE(loc_id.is_implicit());
    EXPECT_THAT(loc_id.node_id(), node_id);

    loc_id = loc_id.ToImplicit();
    EXPECT_TRUE(loc_id.has_value());
    EXPECT_TRUE(loc_id.is_node_id());
    EXPECT_FALSE(loc_id.is_import_ir_inst_id());
    EXPECT_TRUE(loc_id.is_implicit());
    EXPECT_THAT(loc_id.node_id(), node_id);
  }
}

TEST(IdsTest, LocIdIsImportIRInstId) {
  for (auto index : {0, 1, ImportIRInstId::Max - 2, ImportIRInstId::Max - 1}) {
    SCOPED_TRACE(llvm::formatv("Index: {0}", index));
    ImportIRInstId import_ir_inst_id(index);
    LocId loc_id = import_ir_inst_id;
    EXPECT_TRUE(loc_id.has_value());
    EXPECT_FALSE(loc_id.is_node_id());
    EXPECT_TRUE(loc_id.is_import_ir_inst_id());
    EXPECT_FALSE(loc_id.is_implicit());
    EXPECT_THAT(loc_id.import_ir_inst_id(), import_ir_inst_id);
  }
}

}  // namespace
}  // namespace Carbon::SemIR
