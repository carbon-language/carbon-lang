// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <libprotobuf_mutator/src/libfuzzer/libfuzzer_macro.h>

#include "explorer/fuzzing/fuzzer_util.h"

DEFINE_TEXT_PROTO_FUZZER(const Carbon::Fuzzing::Carbon& input) {
  Carbon::ParseAndExecute(input.compilation_unit());
}
