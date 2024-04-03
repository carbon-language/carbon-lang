// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_or.carbon

fn Or(a: i32, b: i32) -> i32 = "int.or";

var arr: [i32; Or(12, 10)];
let arr_p: [i32; 14]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Or(a, b);
}

// CHECK:STDOUT: --- int_or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 14 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 14] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Or = %Or
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Or: <function> = fn_decl @Or [template] {
// CHECK:STDOUT:     %a.loc2_7.1: i32 = param a
// CHECK:STDOUT:     @Or.%a: i32 = bind_name a, %a.loc2_7.1
// CHECK:STDOUT:     %b.loc2_15.1: i32 = param b
// CHECK:STDOUT:     @Or.%b: i32 = bind_name b, %b.loc2_15.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Or.ref: <function> = name_ref Or, %Or [template = %Or]
// CHECK:STDOUT:   %.loc4_19: i32 = int_literal 12 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_23: i32 = int_literal 10 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_18: init i32 = call %Or.ref(%.loc4_19, %.loc4_23) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_26: type = array_type %.loc4_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 14] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 14] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 14 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_20: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_21: type = ptr_type [i32; 14] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 14] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_25: [i32; 14]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 14]* = bind_name arr_p, %.loc5_25
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or(%a: i32, %b: i32) -> i32 = "int.or";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Or.ref: <function> = name_ref Or, file.%Or [template = file.%Or]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_12.1: init i32 = call %Or.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_18: i32 = value_of_initializer %.loc8_12.1
// CHECK:STDOUT:   %.loc8_12.2: i32 = converted %.loc8_12.1, %.loc8_18
// CHECK:STDOUT:   return %.loc8_12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
