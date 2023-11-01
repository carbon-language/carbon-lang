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

  // Matches the ID of an instruction. The numbers may change because of builtin
  // cross-references, so this code is only doing loose structural checks.
  auto integer_id = Yaml::Scalar(MatchesRegex(R"(int\d+)"));
  auto inst_id = Yaml::Scalar(MatchesRegex(R"(inst\+\d+)"));
  auto inst_builtin = Yaml::Scalar(MatchesRegex(R"(inst\w+)"));
  auto type_id = Yaml::Scalar(MatchesRegex(R"(type\d+)"));
  auto type_builtin = Pair(
      type_id, Yaml::Mapping(ElementsAre(Pair("inst", inst_builtin),
                                         Pair("value_rep", Yaml::Mapping(_)))));

  auto file = Yaml::Mapping(ElementsAre(
      Pair("cross_reference_irs_size", "1"),
      Pair("functions", Yaml::Mapping(SizeIs(1))),
      Pair("classes", Yaml::Mapping(SizeIs(0))),
      Pair("types", Yaml::Mapping(Each(type_builtin))),
      Pair("type_blocks", Yaml::Mapping(IsEmpty())),
      Pair("insts",
           Yaml::Mapping(AllOf(
               Each(Key(inst_id)),
               // kind is required, other parts are optional.
               Each(Pair(_, Yaml::Mapping(Contains(Pair("kind", _))))),
               // A 0-arg instruction.
               Contains(
                   Pair(_, Yaml::Mapping(ElementsAre(Pair("kind", "Return"))))),
               // A 1-arg instruction.
               Contains(Pair(
                   _, Yaml::Mapping(ElementsAre(Pair("kind", "IntegerLiteral"),
                                                Pair("arg0", integer_id),
                                                Pair("type", type_id))))),
               // A 2-arg instruction.
               Contains(Pair(
                   _, Yaml::Mapping(ElementsAre(Pair("kind", "Assign"),
                                                Pair("arg0", inst_id),
                                                Pair("arg1", inst_id)))))))),
      // This production has only two instruction blocks.
      Pair("inst_blocks",
           Yaml::Mapping(ElementsAre(
               Pair("block0", Yaml::Mapping(IsEmpty())),
               Pair("block1", Yaml::Mapping(Each(Pair(_, inst_id)))),
               Pair("block2", Yaml::Mapping(Each(Pair(_, inst_id)))))))));

  auto root = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("filename", "test.carbon"), Pair("sem_ir", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()), IsYaml(root));
}

}  // namespace
}  // namespace Carbon::SemIR
