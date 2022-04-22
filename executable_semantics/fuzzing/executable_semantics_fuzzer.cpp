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
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

std::string GetProgramPath() {
  std::string program_name;
#if defined(OS_MACOSX)
#else
  std::error_code error;
  program_name = std::filesystem::canonical("/proc/self/exe", error);
  CHECK(error.value() == 0);
#endif
  llvm::errs() << "### program name=" << program_name << "\n";
  return program_name;
}

std::string GetRunfilesDir() {
  const std::string program_name;
  const std::string runfiles_dir = GetProgramPath() + ".runfiles";
  CHECK(std::filesystem::exists(runfiles_dir));
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
