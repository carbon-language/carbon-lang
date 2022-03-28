// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <google/protobuf/text_format.h>

#include "common/fuzzing/carbon.pb.h"
#include "common/fuzzing/proto_to_carbon.h"
#include "executable_semantics/fuzzing/fuzzer_util.h"
#include "executable_semantics/interpreter/exec_program.h"
#include "executable_semantics/prelude.h"
#include "executable_semantics/syntax/parse.h"
#include "libprotobuf_mutator/src/libfuzzer/libfuzzer_macro.h"
#include "llvm/Support/raw_ostream.h"

namespace Carbon {

// Parses and executes a fuzzer-generated program.
// Takes `compilation_unit` by value to allow modifications like adding
// `Main()`.
void ParseAndExecute(Fuzzing::CompilationUnit compilation_unit) {
  MaybeAddMain(compilation_unit);
  const std::string source = Carbon::ProtoToCarbon(compilation_unit);

  Carbon::Arena arena;
  ErrorOr<AST> ast = Carbon::ParseFromString(&arena, "Fuzzer.carbon", source,
                                             /*trace=*/false);
  if (!ast.ok()) {
    llvm::errs() << "Parsing failed: " << ast.error().message() << "\n";
    return;
  }
  AddPrelude(DefaultPreludeFilename, &arena, &ast->declarations);
  const ErrorOr<int> result =
      Carbon::ExecProgram(&arena, *ast, /*trace=*/false);
  if (!result.ok()) {
    llvm::errs() << "Execution failed: " << result.error().message() << "\n";
    return;
  }
  llvm::outs() << "Executed OK: " << *result;
}

}  // namespace Carbon

DEFINE_BINARY_PROTO_FUZZER(const Carbon::Fuzzing::Carbon& input) {
  Carbon::ParseAndExecute(input.compilation_unit());
}
