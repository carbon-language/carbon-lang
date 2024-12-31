// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/eq.carbon

// --- int_eq.carbon

fn Eq(a: i32, b: i32) -> bool = "int.eq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Eq(1, 1) then True else False);
  false_ as (if Eq(1, 2) then True else False);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> bool {
  return Eq(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+3]]:1: error: invalid signature for builtin function "int.eq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";
