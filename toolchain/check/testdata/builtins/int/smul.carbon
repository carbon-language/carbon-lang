// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smul.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smul.carbon

// --- int_mul.carbon

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

var arr: [i32; Mul(3, 2)];
let arr_p: [i32; 6]* = &arr;

fn RuntimeCallIsValid(a: i32, b: i32) -> i32 {
  return Mul(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Mul(a: i32, b: i32) -> i32 = "int.smul";

let a: i32 = Mul(0x7FFF, 0x10000);
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+3]]:14: error: integer overflow in calculation 32768 * 65536 [CompileTimeIntegerOverflow]
// CHECK:STDERR: let b: i32 = Mul(0x8000, 0x10000);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~
let b: i32 = Mul(0x8000, 0x10000);
