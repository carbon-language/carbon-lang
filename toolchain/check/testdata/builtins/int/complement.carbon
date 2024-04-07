// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_complement.carbon

fn Complement(a: i32) -> i32 = "int.complement";
fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(Complement(0x123456), 0xFFFFFF)];
let arr_p: [i32; 0xEDCBA9]* = &arr;

fn RuntimeCall(a: i32) -> i32 {
  return Complement(a);
}

// CHECK:STDOUT: --- int_complement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1193046 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal -1193047 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 16777215 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 15584169 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type [i32; 15584169] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Complement = %Complement
// CHECK:STDOUT:     .And = %And
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Complement: <function> = fn_decl @Complement [template] {
// CHECK:STDOUT:     %a.loc2_15.1: i32 = param a
// CHECK:STDOUT:     @Complement.%a: i32 = bind_name a, %a.loc2_15.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And: <function> = fn_decl @And [template] {
// CHECK:STDOUT:     %a.loc3_8.1: i32 = param a
// CHECK:STDOUT:     @And.%a: i32 = bind_name a, %a.loc3_8.1
// CHECK:STDOUT:     %b.loc3_16.1: i32 = param b
// CHECK:STDOUT:     @And.%b: i32 = bind_name b, %b.loc3_16.1
// CHECK:STDOUT:     %return.var.loc3: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.ref: <function> = name_ref And, %And [template = %And]
// CHECK:STDOUT:   %Complement.ref: <function> = name_ref Complement, %Complement [template = %Complement]
// CHECK:STDOUT:   %.loc5_31: i32 = int_literal 1193046 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_30.1: init i32 = call %Complement.ref(%.loc5_31) [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_42: i32 = int_literal 16777215 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_19.1: i32 = value_of_initializer %.loc5_30.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_30.2: i32 = converted %.loc5_30.1, %.loc5_19.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_19.2: init i32 = call %And.ref(%.loc5_30.2, %.loc5_42) [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_51: type = array_type %.loc5_19.2, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref [i32; 15584169] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 15584169] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc6_18: i32 = int_literal 15584169 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_26: type = array_type %.loc6_18, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc6_27: type = ptr_type [i32; 15584169] [template = constants.%.6]
// CHECK:STDOUT:   %arr.ref: ref [i32; 15584169] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc6_31: [i32; 15584169]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 15584169]* = bind_name arr_p, %.loc6_31
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc8_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc8_16.1
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Complement(%a: i32) -> i32 = "int.complement";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And(%a: i32, %b: i32) -> i32 = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Complement.ref: <function> = name_ref Complement, file.%Complement [template = file.%Complement]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc9_20.1: init i32 = call %Complement.ref(%a.ref)
// CHECK:STDOUT:   %.loc9_23: i32 = value_of_initializer %.loc9_20.1
// CHECK:STDOUT:   %.loc9_20.2: i32 = converted %.loc9_20.1, %.loc9_23
// CHECK:STDOUT:   return %.loc9_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
