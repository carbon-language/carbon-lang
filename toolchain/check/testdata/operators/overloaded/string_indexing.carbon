// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/string_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/string_indexing.carbon


// --- test_string_indexing.carbon


import Core library "io";
import Core library "range";

fn PrintStr(msg: str) {
  for (i: i32 in Core.Range(msg.Size() as i32)) {
    Core.PrintChar(msg[i]);
  }
}

fn Run() {
  PrintStr("Hello World!\n");
}
