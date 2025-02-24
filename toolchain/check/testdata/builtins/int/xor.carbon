// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/xor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/xor.carbon

// --- int_xor.carbon

fn Xor(a: i32, b: i32) -> i32 = "int.xor";

var arr: array(i32, Xor(12, 10));
let arr_p: array(i32, 6)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Xor(a, b);
}
