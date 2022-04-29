// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/fuzzing/fuzzer_util.h"

#include <filesystem>

#include "common/check.h"
#include "common/fuzzing/proto_to_carbon.h"
#include "executable_semantics/interpreter/exec_program.h"
#include "executable_semantics/syntax/parse.h"
#include "executable_semantics/syntax/prelude.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace Carbon {

// Appended to fuzzer-generated Carbon source when the source is missing
// `Main()` definition, to prevent early error return in semantic analysis.
static constexpr char EmptyMain[] = R"(
fn Main() -> i32 {
  return 0;
}
)";

auto ProtoToCarbonWithMain(const Fuzzing::CompilationUnit& compilation_unit)
    -> std::string {
  const bool has_main = std::any_of(
      compilation_unit.declarations().begin(),
      compilation_unit.declarations().end(),
      [](const Fuzzing::Declaration& decl) {
        return decl.kind_case() == Fuzzing::Declaration::kFunction &&
               decl.function().name() == "Main";
      });
  return Carbon::ProtoToCarbon(compilation_unit) + (has_main ? "" : EmptyMain);
}

auto GetRunfilesFile(const std::string& path) -> std::string {
  using bazel::tools::cpp::runfiles::Runfiles;
  std::string error;
  std::unique_ptr<Runfiles> runfiles(Runfiles::Create(
      llvm::sys::fs::getMainExecutable(nullptr, nullptr),  // argv0,
      &error));
  CHECK(runfiles != nullptr) << error;
  return runfiles->Rlocation(path);
}

void ParseAndExecute(const Fuzzing::CompilationUnit& compilation_unit) {
  const std::string source = ProtoToCarbonWithMain(compilation_unit);

  Arena arena;
  ErrorOr<AST> ast = ParseFromString(&arena, "Fuzzer.carbon", source,
                                     /*parser_debug=*/false);
  if (!ast.ok()) {
    llvm::errs() << "Parsing failed: " << ast.error().message() << "\n";
    return;
  }
  llvm::errs() << "current_dir=" << std::filesystem::current_path() << "\n";
  AddPrelude(GetRunfilesFile("carbon/executable_semantics/data/prelude.carbon"),
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
