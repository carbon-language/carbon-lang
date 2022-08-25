// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <libprotobuf_mutator/src/libfuzzer/libfuzzer_macro.h>

#include "common/error.h"
#include "explorer/fuzzing/fuzzer_util.h"
#include "llvm/Support/raw_ostream.h"

DEFINE_TEXT_PROTO_FUZZER(const Carbon::Fuzzing::Carbon& input) {
  const auto result = Carbon::ParseAndExecute(input.compilation_unit());
  if (result.ok()) {
    llvm::outs() << "Executed OK: " << *result << "\n";
  } else {
    llvm::errs() << "Execution failed: " << result.error() << "\n";
  }
}
