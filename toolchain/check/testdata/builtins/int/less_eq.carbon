// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/less_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/less_eq.carbon

// --- int_less_eq.carbon

library "[[@TEST_NAME]]";

fn LessEq(a: i32, b: i32) -> bool = "int.less_eq";
fn Negate(a: i32) -> i32 = "int.snegate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if LessEq(1, 2) then True else False);
  true_ as (if LessEq(1, 1) then True else False);
  false_ as (if LessEq(1, 0) then True else False);
  true_ as (if LessEq(Negate(1), 0) then True else False);
  false_ as (if LessEq(0, Negate(1)) then True else False);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> bool {
  return LessEq(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(LessEq(5, 5)) as Expect(true);
  Test(LessEq(5, 6)) as Expect(true);
  Test(LessEq(6, 5)) as Expect(false);
  Test(LessEq(-1, -1)) as Expect(true);
  Test(LessEq(-1, 1)) as Expect(true);
  Test(LessEq(1, -1)) as Expect(false);
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn LessEq(a: Core.IntLiteral(), b: i32) -> bool = "int.less_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(LessEq(5, 5)) as Expect(true);
  Test(LessEq(5, 6)) as Expect(true);
  Test(LessEq(6, 5)) as Expect(false);
  Test(LessEq(-1, -1)) as Expect(true);
  Test(LessEq(-1, 1)) as Expect(true);
  Test(LessEq(1, -1)) as Expect(false);
}

// --- fail_runtime_literal.carbon

library "[[@TEST_NAME]]";

fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";

fn Test(n: Core.IntLiteral()) {
  // OK
  LessEq(1, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   LessEq(n, 1);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-8]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  LessEq(n, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   LessEq(1, n);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-16]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  LessEq(1, n);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   LessEq(n, n);
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-24]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn LessEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.less_eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  LessEq(n, n);
}
