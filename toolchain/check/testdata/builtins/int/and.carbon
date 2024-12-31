// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/and.carbon

// --- int_and.carbon

fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(12, 10)];
let arr_p: [i32; 8]* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return And(a, b);
}
