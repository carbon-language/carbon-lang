// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "testing/util/test_raw_ostream.h"
#include "toolchain/common/yaml_test_helpers.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::MatchesRegex;
using ::testing::Pair;

TEST(SemanticsIRTest, YAML) {
  llvm::vfs::InMemoryFileSystem fs;
  CARBON_CHECK(fs.addFile("test.carbon", /*ModificationTime=*/0,
                          llvm::MemoryBuffer::getMemBuffer("var x: i32 = 0;")));
  TestRawOstream print_stream;
  Driver d(fs, print_stream, llvm::errs());
  d.RunFullCommand({"dump", "raw-semantics-ir", "test.carbon"});

  // Matches the ID of a node. The numbers may change because of builtin
  // cross-references, so this code is only doing loose structural checks.
  auto node_id = Yaml::Scalar(MatchesRegex(R"(node\+\d+)"));
  auto node_builtin = Yaml::Scalar(MatchesRegex(R"(node\w+)"));
  auto type_id = Yaml::Scalar(MatchesRegex(R"(type\d+)"));

  EXPECT_THAT(
      Yaml::Value::FromText(print_stream.TakeStr()),
      ElementsAre(Yaml::Mapping(ElementsAre(
          Pair("cross_reference_irs_size", "1"),
          Pair("functions", Yaml::Sequence(IsEmpty())),
          Pair("integer_literals", Yaml::Sequence(ElementsAre("0"))),
          Pair("real_literals", Yaml::Sequence(IsEmpty())),
          Pair("strings", Yaml::Sequence(ElementsAre("x"))),
          Pair("types", Yaml::Sequence(ElementsAre(node_builtin))),
          Pair("type_blocks", Yaml::Sequence(IsEmpty())),
          Pair("nodes",
               Yaml::Sequence(AllOf(
                   // kind is required, other parts are optional.
                   Each(Yaml::Mapping(Contains(Pair("kind", _)))),
                   // A 0-arg node.
                   Contains(Yaml::Mapping(ElementsAre(
                       Pair("kind", "VarStorage"), Pair("type", type_id)))),
                   // A 1-arg node.
                   Contains(Yaml::Mapping(ElementsAre(
                       Pair("kind", "IntegerLiteral"), Pair("arg0", "int0"),
                       Pair("type", type_id)))),
                   // A 2-arg node.
                   Contains(Yaml::Mapping(ElementsAre(
                       Pair("kind", "BindName"), Pair("arg0", "str0"),
                       Pair("arg1", node_id), Pair("type", type_id))))))),
          // This production has only one node block.
          Pair("node_blocks",
               Yaml::Sequence(ElementsAre(Yaml::Sequence(IsEmpty()),
                                          Yaml::Sequence(Each(node_id)))))))));
}

}  // namespace
}  // namespace Carbon::Testing
