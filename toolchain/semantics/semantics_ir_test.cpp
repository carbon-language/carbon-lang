// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <forward_list>

#include "toolchain/common/yaml_test_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lexer/tokenized_buffer.h"

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
  DiagnosticConsumer& consumer = ConsoleDiagnosticConsumer();
  llvm::Expected<SourceBuffer> source =
      SourceBuffer::CreateFromText("var x: i32 = 0;");
  TokenizedBuffer tokens = TokenizedBuffer::Lex(*source, consumer);
  ParseTree parse_tree =
      ParseTree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
  SemanticsIR builtin_ir = SemanticsIR::MakeBuiltinIR();
  SemanticsIR semantics_ir = SemanticsIR::MakeFromParseTree(
      builtin_ir, tokens, parse_tree, consumer, /*vlog_stream=*/nullptr);

  std::string print_output;
  llvm::raw_string_ostream print_stream(print_output);
  semantics_ir.Print(print_stream);
  print_stream.flush();

  // Matches the ID of a node. The numbers may change because of builtin
  // cross-references, so this code is only doing loose structural checks.
  auto node_id = Yaml::Scalar(MatchesRegex(R"(node\+\d+)"));
  auto node_builtin = Yaml::Scalar(MatchesRegex(R"(node\w+)"));

  EXPECT_THAT(
      Yaml::Value::FromText(print_output),
      ElementsAre(Yaml::Mapping(ElementsAre(
          Pair("cross_reference_irs_size", "1"),
          Pair("calls", Yaml::Sequence(IsEmpty())),
          Pair("callables", Yaml::Sequence(IsEmpty())),
          Pair("integer_literals", Yaml::Sequence(ElementsAre("0"))),
          Pair("real_literals", Yaml::Sequence(IsEmpty())),
          Pair("strings", Yaml::Sequence(ElementsAre("x"))),
          Pair(
              "nodes",
              Yaml::Sequence(AllOf(
                  // kind is required, other parts are optional.
                  Each(Yaml::Mapping(Contains(Pair("kind", _)))),
                  // A 0-arg node.
                  Contains(Yaml::Mapping(ElementsAre(
                      Pair("kind", "VarStorage"), Pair("type", node_builtin)))),
                  // A 1-arg node.
                  Contains(Yaml::Mapping(ElementsAre(
                      Pair("kind", "IntegerLiteral"), Pair("arg0", "int0"),
                      Pair("type", node_builtin)))),
                  // A 2-arg node.
                  Contains(Yaml::Mapping(ElementsAre(
                      Pair("kind", "BindName"), Pair("arg0", "str0"),
                      Pair("arg1", node_id), Pair("type", node_builtin))))))),
          // This production has only one node block.
          Pair("node_blocks",
               Yaml::Sequence(ElementsAre(Yaml::Sequence(IsEmpty()),
                                          Yaml::Sequence(Each(node_id)))))))));
}

}  // namespace
}  // namespace Carbon::Testing
