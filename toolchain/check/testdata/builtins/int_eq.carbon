// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_eq.carbon

fn Eq(a: i32, b: i32) -> bool = "int.eq";

// TODO: Use a test that will fail to compile if we get the wrong value.
let b_true: bool = Eq(1, 1);
let b_false: bool = Eq(1, 2);

fn RuntimeCall(a: i32, b: i32) -> bool {
  return Eq(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+3]]:1: ERROR: Invalid signature for builtin function "int.eq".
// CHECK:STDERR: fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn WrongResult(a: i32, b: i32) -> i32 = "int.eq";

// CHECK:STDOUT: --- int_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Eq = %Eq
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eq: <function> = fn_decl @Eq [template] {
// CHECK:STDOUT:     %a.loc2_7.1: i32 = param a
// CHECK:STDOUT:     @Eq.%a: i32 = bind_name a, %a.loc2_7.1
// CHECK:STDOUT:     %b.loc2_15.1: i32 = param b
// CHECK:STDOUT:     @Eq.%b: i32 = bind_name b, %b.loc2_15.1
// CHECK:STDOUT:     %return.var.loc2: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Eq.ref.loc5: <function> = name_ref Eq, %Eq [template = %Eq]
// CHECK:STDOUT:   %.loc5_23: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_26: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_22.1: init bool = call %Eq.ref.loc5(%.loc5_23, %.loc5_26) [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_28: bool = value_of_initializer %.loc5_22.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_22.2: bool = converted %.loc5_22.1, %.loc5_28 [template = constants.%.2]
// CHECK:STDOUT:   %b_true: bool = bind_name b_true, %.loc5_22.2
// CHECK:STDOUT:   %Eq.ref.loc6: <function> = name_ref Eq, %Eq [template = %Eq]
// CHECK:STDOUT:   %.loc6_24: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_27: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc6_23.1: init bool = call %Eq.ref.loc6(%.loc6_24, %.loc6_27) [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_29: bool = value_of_initializer %.loc6_23.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_23.2: bool = converted %.loc6_23.1, %.loc6_29 [template = constants.%.4]
// CHECK:STDOUT:   %b_false: bool = bind_name b_false, %.loc6_23.2
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc8_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc8_16.1
// CHECK:STDOUT:     %b.loc8_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc8_24.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Eq(%a: i32, %b: i32) -> bool = "int.eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Eq.ref: <function> = name_ref Eq, file.%Eq [template = file.%Eq]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc9_12.1: init bool = call %Eq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc9_18: bool = value_of_initializer %.loc9_12.1
// CHECK:STDOUT:   %.loc9_12.2: bool = converted %.loc9_12.1, %.loc9_18
// CHECK:STDOUT:   return %.loc9_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .WrongResult = %WrongResult
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongResult: <function> = fn_decl @WrongResult [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @WrongResult.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @WrongResult.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WrongResult(%a: i32, %b: i32) -> i32;
// CHECK:STDOUT:
