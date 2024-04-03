// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_xor.carbon

fn Xor(a: i32, b: i32) -> i32 = "int.xor";

var arr: [i32; Xor(12, 10)];
let arr_p: [i32; 6]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Xor(a, b);
}

// CHECK:STDOUT: --- int_xor.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 6] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Xor = %Xor
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Xor: <function> = fn_decl @Xor [template] {
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Xor.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Xor.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Xor.ref: <function> = name_ref Xor, %Xor [template = %Xor]
// CHECK:STDOUT:   %.loc4_20: i32 = int_literal 12 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_24: i32 = int_literal 10 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_19: init i32 = call %Xor.ref(%.loc4_20, %.loc4_24) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_27: type = array_type %.loc4_19, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 6] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 6] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 6 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type [i32; 6] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 6] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_24: [i32; 6]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 6]* = bind_name arr_p, %.loc5_24
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Xor(%a: i32, %b: i32) -> i32 = "int.xor";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Xor.ref: <function> = name_ref Xor, file.%Xor [template = file.%Xor]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_13.1: init i32 = call %Xor.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19: i32 = value_of_initializer %.loc8_13.1
// CHECK:STDOUT:   %.loc8_13.2: i32 = converted %.loc8_13.1, %.loc8_19
// CHECK:STDOUT:   return %.loc8_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
