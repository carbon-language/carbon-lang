// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_neq.carbon

fn Neq(a: i32, b: i32) -> bool = "int.neq";

// TODO: Use a test that will fail to compile if we get the wrong value.
let b_true: bool = Neq(1, 2);
let b_false: bool = Neq(1, 1);

fn RuntimeCall(a: i32, b: i32) -> bool {
  return Neq(a, b);
}

// CHECK:STDOUT: --- int_neq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: bool = bool_literal true [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Neq = %Neq
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Neq: <function> = fn_decl @Neq [template] {
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Neq.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Neq.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %return.var.loc2: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Neq.ref.loc5: <function> = name_ref Neq, %Neq [template = %Neq]
// CHECK:STDOUT:   %.loc5_24: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_27: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_23.1: init bool = call %Neq.ref.loc5(%.loc5_24, %.loc5_27) [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_29: bool = value_of_initializer %.loc5_23.1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_23.2: bool = converted %.loc5_23.1, %.loc5_29 [template = constants.%.3]
// CHECK:STDOUT:   %b_true: bool = bind_name b_true, %.loc5_23.2
// CHECK:STDOUT:   %Neq.ref.loc6: <function> = name_ref Neq, %Neq [template = %Neq]
// CHECK:STDOUT:   %.loc6_25: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_28: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_24.1: init bool = call %Neq.ref.loc6(%.loc6_25, %.loc6_28) [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_30: bool = value_of_initializer %.loc6_24.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_24.2: bool = converted %.loc6_24.1, %.loc6_30 [template = constants.%.4]
// CHECK:STDOUT:   %b_false: bool = bind_name b_false, %.loc6_24.2
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc8_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc8_16.1
// CHECK:STDOUT:     %b.loc8_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc8_24.1
// CHECK:STDOUT:     %return.var.loc8: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Neq(%a: i32, %b: i32) -> bool = "int.neq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Neq.ref: <function> = name_ref Neq, file.%Neq [template = file.%Neq]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc9_13.1: init bool = call %Neq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc9_19: bool = value_of_initializer %.loc9_13.1
// CHECK:STDOUT:   %.loc9_13.2: bool = converted %.loc9_13.1, %.loc9_19
// CHECK:STDOUT:   return %.loc9_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
