// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/fuzzing/fuzzer_util.h"

#include "common/check.h"
#include "common/fuzzing/proto_to_carbon.h"
#include "explorer/interpreter/exec_program.h"
#include "explorer/syntax/parse.h"
#include "explorer/syntax/prelude.h"
#include "llvm/Support/raw_ostream.h"

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

void ParseAndExecute(const Fuzzing::CompilationUnit& compilation_unit) {
  const std::string source = ProtoToCarbonWithMain(compilation_unit);

  Arena arena;
  ErrorOr<AST> ast = ParseFromString(&arena, "Fuzzer.carbon", source,
                                     /*parser_debug=*/false);
  if (!ast.ok()) {
    llvm::errs() << "Parsing failed: " << ast.error().message() << "\n";
    return;
  }
  AddPrelude("explorer/data/prelude.carbon", &arena, &ast->declarations);
  const ErrorOr<int> result =
      ExecProgram(&arena, *ast, /*trace_stream=*/std::nullopt);
  if (!result.ok()) {
    llvm::errs() << "Execution failed: " << result.error().message() << "\n";
    return;
  }
  llvm::outs() << "Executed OK: " << *result << "\n";
}

}  // namespace Carbon
