// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <google/protobuf/text_format.h>
#include <libprotobuf_mutator/src/libfuzzer/libfuzzer_macro.h>

#include <filesystem>

#include "common/fuzzing/carbon.pb.h"
#include "executable_semantics/fuzzing/fuzzer_util.h"
#include "executable_semantics/interpreter/exec_program.h"
#include "executable_semantics/syntax/parse.h"
#include "executable_semantics/syntax/prelude.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Determines runfiles dir to use.
static auto GetRunfilesDir() -> std::string {
  const char* test_src_dir = getenv("TEST_SRCDIR");
  std::string runfiles_dir =
      test_src_dir != nullptr
          ? test_src_dir
          : llvm::sys::fs::getMainExecutable(nullptr, nullptr) + ".runfiles";
  CHECK(std::filesystem::exists(runfiles_dir))
      << runfiles_dir << " doesn't exist";
  return runfiles_dir;
}

// Parses and executes a fuzzer-generated program.
void ParseAndExecute(const Fuzzing::CompilationUnit& compilation_unit) {
  const std::string source = ProtoToCarbonWithMain(compilation_unit);

  Arena arena;
  ErrorOr<AST> ast = ParseFromString(&arena, "Fuzzer.carbon", source,
                                     /*parser_debug=*/false);
  if (!ast.ok()) {
    llvm::errs() << "Parsing failed: " << ast.error().message() << "\n";
    return;
  }
  AddPrelude(
      GetRunfilesDir() + "/carbon/executable_semantics/data/prelude.carbon",
      &arena, &ast->declarations);
  const ErrorOr<int> result =
      ExecProgram(&arena, *ast, /*trace_stream=*/std::nullopt);
  if (!result.ok()) {
    llvm::errs() << "Execution failed: " << result.error().message() << "\n";
    return;
  }
  llvm::outs() << "Executed OK: " << *result << "\n";
}

}  // namespace Carbon

DEFINE_TEXT_PROTO_FUZZER(const Carbon::Fuzzing::Carbon& input) {
  Carbon::ParseAndExecute(input.compilation_unit());
}
