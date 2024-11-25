// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "testing/base/global_exe_path.h"
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
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::MatchesRegex;
using ::testing::Pair;
using ::testing::SizeIs;

namespace Yaml = ::Carbon::Testing::Yaml;

TEST(SemIRTest, YAML) {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  CARBON_CHECK(fs->addFile(
      "test.carbon", /*ModificationTime=*/0,
      llvm::MemoryBuffer::getMemBuffer("fn F() { var x: () = (); return; }")));
  const auto install_paths =
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath());
  TestRawOstream print_stream;
  Driver d(fs, &install_paths, print_stream, llvm::errs());
  auto run_result =
      d.RunCommand({"compile", "--no-prelude-import", "--phase=check",
                    "--dump-raw-sem-ir", "test.carbon"});
  EXPECT_TRUE(run_result.success);

  // Matches the ID of an instruction. Instruction counts may change as various
  // support changes, so this code is only doing loose structural checks.
  auto type_block_id = Yaml::Scalar(MatchesRegex(R"(type_block\d+)"));
  auto inst_id = Yaml::Scalar(MatchesRegex(R"(inst\+\d+)"));
  auto constant_id =
      Yaml::Scalar(MatchesRegex(R"(templateConstant\(inst(\w+|\+\d+)\))"));
  auto inst_builtin = Yaml::Scalar(MatchesRegex(R"(inst\w+)"));
  auto type_id = Yaml::Scalar(MatchesRegex(R"(type(\w+|\(inst(\w+|\+\d+)\)))"));
  auto type_builtin = Pair(type_id, Yaml::Mapping(_));

  auto file = Yaml::Mapping(ElementsAre(
      Pair("import_irs", Yaml::Mapping(SizeIs(1))),
      Pair("import_ir_insts", Yaml::Mapping(SizeIs(0))),
      Pair("name_scopes", Yaml::Mapping(SizeIs(1))),
      Pair("entity_names", Yaml::Mapping(SizeIs(1))),
      Pair("functions", Yaml::Mapping(SizeIs(1))),
      Pair("classes", Yaml::Mapping(SizeIs(0))),
      Pair("generics", Yaml::Mapping(SizeIs(0))),
      Pair("specifics", Yaml::Mapping(SizeIs(0))),
      Pair("struct_type_fields", Yaml::Mapping(SizeIs(1))),
      Pair("types", Yaml::Mapping(Each(type_builtin))),
      Pair("type_blocks", Yaml::Mapping(SizeIs(Ge(1)))),
      Pair("insts",
           Yaml::Mapping(AllOf(
               Each(Key(inst_id)),
               // kind is required, other parts are optional.
               Each(Pair(_, Yaml::Mapping(Contains(Pair("kind", _))))),
               // A 0-arg instruction.
               Contains(
                   Pair(_, Yaml::Mapping(ElementsAre(Pair("kind", "Return"))))),
               // A 1-arg instruction.
               Contains(Pair(_, Yaml::Mapping(ElementsAre(
                                    Pair("kind", "TupleType"),
                                    Pair("arg0", type_block_id),
                                    Pair("type", "typeTypeType"))))),
               // A 2-arg instruction.
               Contains(Pair(
                   _, Yaml::Mapping(ElementsAre(Pair("kind", "Assign"),
                                                Pair("arg0", inst_id),
                                                Pair("arg1", inst_id)))))))),
      Pair("constant_values",
           Yaml::Mapping(AllOf(Each(Pair(inst_id, constant_id))))),
      Pair("symbolic_constants", Yaml::Mapping(SizeIs(0))),
      // This production has only two instruction blocks.
      Pair("inst_blocks",
           Yaml::Mapping(ElementsAre(
               Pair("empty", Yaml::Mapping(IsEmpty())),
               Pair("exports", Yaml::Mapping(Each(Pair(_, inst_id)))),
               Pair("import_refs", Yaml::Mapping(IsEmpty())),
               Pair("global_init", Yaml::Mapping(IsEmpty())),
               Pair("block4", Yaml::Mapping(Each(Pair(_, inst_id)))),
               Pair("block5", Yaml::Mapping(Each(Pair(_, inst_id)))))))));

  auto root = Yaml::Sequence(ElementsAre(Yaml::Mapping(
      ElementsAre(Pair("filename", "test.carbon"), Pair("sem_ir", file)))));

  EXPECT_THAT(Yaml::Value::FromText(print_stream.TakeStr()), IsYaml(root));
}

}  // namespace
}  // namespace Carbon::SemIR
