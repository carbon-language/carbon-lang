// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <libfuzzer/libfuzzer_macro.h>

#include "common/error.h"
#include "explorer/fuzzing/fuzzer_util.h"
#include "llvm/Support/raw_ostream.h"

DEFINE_TEXT_PROTO_FUZZER(const Carbon::Fuzzing::Carbon& input) {
  // Only verifying it doesn't crash.
  (void)Carbon::Testing::ParseAndExecuteProto(input);
}
