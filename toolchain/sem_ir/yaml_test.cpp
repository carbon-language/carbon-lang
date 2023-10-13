// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "testing/base/test_raw_ostream.h"
#include "toolchain/driver/driver.h"
#include "toolchain/testing/yaml_test_helpers.h"

namespace Carbon::SemIR {
namespace {

using ::Carbon::Testing::TestRawOstream;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::MatchesRegex;
using ::testing::Pair;
using ::testing::SizeIs;

namespace Yaml = ::Carbon::Testing::Yaml;

TEST(SemIRTest, YAML) {
  llvm::vfs::InMemoryFileSystem fs;
  CARBON_CHECK(fs.addFile(
      "test.carbon", /*ModificationTime=*/0,
      llvm::MemoryBuffer::getMemBuffer("fn F() { var x: i32 = 0; return; }")));
  TestRawOstream print_stream;
  Driver d(fs, print_stream, llvm::errs());
  d.RunCommand(
      {"compile", "--phase=check", "--dump-raw-sem-ir", "test.carbon"});

  // Matches the ID of a node. The numbers may change because of builtin
  // cross-references, so this code is only doing loose structural checks.
  auto node_id = Yaml::Scalar(MatchesRegex(R"(node\+\d+)"));
  auto node_builtin = Yaml::Scalar(MatchesRegex(R"(node\w+)"));
  auto type_id = Yaml::Scalar(MatchesRegex(R"(type\d+)"));
  auto type_builtin = Yaml::Mapping(ElementsAre(
      Pair("node", node_builtin), Pair("value_rep", Yaml::Mapping(_))));

  auto file = Yaml::Sequence(ElementsAre(Yaml::Mapping(ElementsAre(
      Pair("cross_reference_irs_size", "1"),
      Pair("functions", Yaml::Sequence(SizeIs(1))),
      Pair("classes", Yaml::Sequence(SizeIs(0))),
      Pair("integers", Yaml::Sequence(ElementsAre("0"))),
      Pair("reals", Yaml::Sequence(IsEmpty())),
      Pair("strings", Yaml::Sequence(ElementsAre("F", "x"))),
      Pair("types", Yaml::Sequence(Each(type_builtin))),
      Pair("type_blocks", Yaml::Sequence(IsEmpty())),
      Pair("nodes",
           Yaml::Sequence(AllOf(
               // kind is required, other parts are optional.
               Each(Yaml::Mapping(Contains(Pair("kind", _)))),
               // A 0-arg node.
               Contains(Yaml::Mapping(ElementsAre(Pair("kind", "Return")))),
               // A 1-arg node.
               Contains(Yaml::Mapping(
                   ElementsAre(Pair("kind", "IntegerLiteral"),
                               Pair("arg0", "int0"), Pair("type", type_id)))),
               // A 2-arg node.
               Contains(Yaml::Mapping(ElementsAre(Pair("kind", "Assign"),
                                                  Pair("arg0", node_id),
                                                  Pair("arg1", node_id))))))),
      // This production has only two node blocks.
      Pair("node_blocks",
           Yaml::Sequence(ElementsAre(Yaml::Sequence(IsEmpty()),
                                      Yaml::Sequence(Each(node_id)),
                                      Yaml::Sequence(Each(node_id)))))))));

  auto root = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("filename", "test.carbon"), Pair("sem_ir", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()),
              IsYaml(ElementsAre(root)));
}

}  // namespace
}  // namespace Carbon::SemIR
