// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/complement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/complement.carbon

// --- int_complement.carbon

library "[[@TEST_NAME]]";

fn Complement(a: i32) -> i32 = "int.complement";
fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: array(i32, And(Complement(0x123456), 0xFFFFFF));
let arr_p: array(i32, 0xEDCBA9)* = &arr;

fn RuntimeCallIsValid(a: i32) -> i32 {
  return Complement(a);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Complement(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.complement";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Complement(0)) as Expect(-1);
  Test(Complement(1)) as Expect(-2);
  Test(Complement(-1)) as Expect(0);
  Test(Complement(-0x7FFF_FFFF_FFFF_FFFF)) as Expect(0x7FFF_FFFF_FFFF_FFFE);
  Test(Complement(-0x8000_0000_0000_0000)) as Expect(0x7FFF_FFFF_FFFF_FFFF);
  Test(Complement(0x7FFF_FFFF_FFFF_FFFF)) as Expect(-0x8000_0000_0000_0000);
  Test(Complement(0x8000_0000_0000_0000)) as Expect(-0x8000_0000_0000_0001);
}

// --- fail_literal_runtime.carbon

library "[[@TEST_NAME]]";

fn Complement(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.complement";

fn F(a: Core.IntLiteral()) -> Core.IntLiteral() {
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE+7]]:10: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   return Complement(a);
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE-6]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Complement(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.complement";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Complement(a);
}
