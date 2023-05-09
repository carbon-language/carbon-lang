// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "testing/file_test/file_test_base.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon::Testing {
namespace {

class SemanticsFileTest : public FileTestBase {
 public:
  explicit SemanticsFileTest(llvm::StringRef path) : FileTestBase(path) {}

  void RunOverFile(llvm::raw_ostream& stdout,
                   llvm::raw_ostream& stderr) override {
    StreamDiagnosticConsumer consumer(stderr);
    llvm::Expected<SourceBuffer> source = SourceBuffer::CreateFromFile(path());
    TokenizedBuffer tokens = TokenizedBuffer::Lex(*source, consumer);
    ParseTree parse_tree =
        ParseTree::Parse(tokens, consumer, /*vlog_stream=*/nullptr);
    SemanticsIR builtin_ir = SemanticsIR::MakeBuiltinIR();
    SemanticsIR semantics_ir = SemanticsIR::MakeFromParseTree(
        builtin_ir, tokens, parse_tree, consumer, /*vlog_stream=*/nullptr);
    semantics_ir.Print(stdout);
  }
};

}  // namespace

auto RegisterFileTests(const std::vector<llvm::StringRef>& paths) -> void {
  SemanticsFileTest::RegisterTests(
      "SemanticsFileTest", paths,
      [](llvm::StringRef path) { return new SemanticsFileTest(path); });
}

}  // namespace Carbon::Testing
