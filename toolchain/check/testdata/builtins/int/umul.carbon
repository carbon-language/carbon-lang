// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/umul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/umul.carbon

// --- int_mul.carbon

fn Mul(a: i32, b: i32) -> i32 = "int.umul";

var arr: array(i32, Mul(3, 2));
let arr_p: array(i32, 6)* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Mul(a, b);
}

// --- overflow.carbon

package Overflow;

fn Mul(a: i32, b: i32) -> i32 = "int.umul";

let a: i32 = Mul(0x7FFF, 0x10000);
let b: i32 = Mul(0x8000, 0x10000);
