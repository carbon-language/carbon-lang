// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/eq.carbon

// --- int_eq.carbon

library "[[@TEST_NAME]]";

fn Eq(a: i32, b: i32) -> bool = "int.eq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Eq(1, 1) then True else False);
  false_ as (if Eq(1, 2) then True else False);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> bool {
  //@dump-sem-ir-begin
  return Eq(a, b);
  //@dump-sem-ir-end
}

// --- fail_bad_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.eq" [InvalidBuiltinSignature]
// CHECK:STDERR: fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(Eq(5, 5)) as Expect(true);
  Test(Eq(5, 6)) as Expect(false);
  Test(Eq(-1, -1)) as Expect(true);
  Test(Eq(-1, 1)) as Expect(false);
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn Eq(a: Core.IntLiteral(), b: i32) -> bool = "int.eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(Eq(5, 5)) as Expect(true);
  Test(Eq(5, 6)) as Expect(false);
  Test(Eq(-1, -1)) as Expect(true);
  Test(Eq(-1, 1)) as Expect(false);
}

// --- fail_runtime_literal.carbon

library "[[@TEST_NAME]]";

fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";

fn Test(n: Core.IntLiteral()) {
  // OK
  Eq(1, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   Eq(n, 1);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-8]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Eq(n, 1);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   Eq(1, n);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-16]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Eq(1, n);
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE+7]]:3: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   Eq(n, n);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_runtime_literal.carbon:[[@LINE-24]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Eq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.eq";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Eq(n, n);
}

// CHECK:STDOUT: --- int_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Eq.type: type = fn_type @Eq [concrete]
// CHECK:STDOUT:   %Eq: %Eq.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallIsValid(%a.param: %i32, %b.param: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Eq.ref: %Eq.type = name_ref Eq, file.%Eq.decl [concrete = constants.%Eq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.eq: init bool = call %Eq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc16_18.1: bool = value_of_initializer %int.eq
// CHECK:STDOUT:   %.loc16_18.2: bool = converted %int.eq, %.loc16_18.1
// CHECK:STDOUT:   return %.loc16_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
