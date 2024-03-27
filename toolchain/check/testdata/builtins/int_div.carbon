// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_div.carbon

fn Div(a: i32, b: i32) -> i32 = "int.div";

var arr: [i32; Div(3, 2)];
let arr_p: [i32; 1]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Div(a, b);
}

// --- fail_overflow.carbon

package FailOverflow api;

fn Div(a: i32, b: i32) -> i32 = "int.div";
fn Sub(a: i32, b: i32) -> i32 = "int.sub";
fn Negate(a: i32) -> i32 = "int.negate";

// -0x7FFF_FFFF / -1 is OK.
let a: i32 = Div(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 / 1 is OK.
let b: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 overflows.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer overflow in calculation -2147483648 / -1.
// CHECK:STDERR: let c: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let c: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

package FailDivByZero api;

fn Div(a: i32, b: i32) -> i32 = "int.div";

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: ERROR: Division by zero.
// CHECK:STDERR: let a: i32 = Div(1, 0);
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let a: i32 = Div(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+3]]:14: ERROR: Division by zero.
// CHECK:STDERR: let b: i32 = Div(0, 0);
// CHECK:STDERR:              ^~~~
let b: i32 = Div(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Div = %Div
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Div: <function> = fn_decl @Div [template] {
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Div.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Div.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Div.ref: <function> = name_ref Div, %Div [template = %Div]
// CHECK:STDOUT:   %.loc4_20: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_23: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_19: init i32 = call %Div.ref(%.loc4_20, %.loc4_23) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 1] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 1] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type [i32; 1] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 1] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_24: [i32; 1]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 1]* = bind_name arr_p, %.loc5_24
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a: i32, %b: i32) -> i32 = "int.div";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref: <function> = name_ref Div, file.%Div [template = file.%Div]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_13.1: init i32 = call %Div.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19: i32 = value_of_initializer %.loc8_13.1
// CHECK:STDOUT:   %.loc8_13.2: i32 = converted %.loc8_13.1, %.loc8_19
// CHECK:STDOUT:   return %.loc8_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal -2147483647 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal -1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Div = %Div
// CHECK:STDOUT:     .Sub = %Sub
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Div: <function> = fn_decl @Div [template] {
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Div.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Div.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     %return.var.loc4: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Sub: <function> = fn_decl @Sub [template] {
// CHECK:STDOUT:     %a.loc5_8.1: i32 = param a
// CHECK:STDOUT:     @Sub.%a: i32 = bind_name a, %a.loc5_8.1
// CHECK:STDOUT:     %b.loc5_16.1: i32 = param b
// CHECK:STDOUT:     @Sub.%b: i32 = bind_name b, %b.loc5_16.1
// CHECK:STDOUT:     %return.var.loc5: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc6_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc6_11.1
// CHECK:STDOUT:     %return.var.loc6: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Div.ref.loc9: <function> = name_ref Div, %Div [template = %Div]
// CHECK:STDOUT:   %Negate.ref.loc9_18: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc9_25: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_24.1: init i32 = call %Negate.ref.loc9_18(%.loc9_25) [template = constants.%.2]
// CHECK:STDOUT:   %Negate.ref.loc9_39: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc9_46: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_45.1: init i32 = call %Negate.ref.loc9_39(%.loc9_46) [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_17.1: i32 = value_of_initializer %.loc9_24.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_24.2: i32 = converted %.loc9_24.1, %.loc9_17.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_17.2: i32 = value_of_initializer %.loc9_45.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_45.2: i32 = converted %.loc9_45.1, %.loc9_17.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_17.3: init i32 = call %Div.ref.loc9(%.loc9_24.2, %.loc9_45.2) [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_49: i32 = value_of_initializer %.loc9_17.3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_17.4: i32 = converted %.loc9_17.3, %.loc9_49 [template = constants.%.1]
// CHECK:STDOUT:   %a.loc9: i32 = bind_name a, %.loc9_17.4
// CHECK:STDOUT:   %Div.ref.loc12: <function> = name_ref Div, %Div [template = %Div]
// CHECK:STDOUT:   %Sub.ref.loc12: <function> = name_ref Sub, %Sub [template = %Sub]
// CHECK:STDOUT:   %Negate.ref.loc12: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc12_29: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_28.1: init i32 = call %Negate.ref.loc12(%.loc12_29) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_43: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_21.1: i32 = value_of_initializer %.loc12_28.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_28.2: i32 = converted %.loc12_28.1, %.loc12_21.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_21.2: init i32 = call %Sub.ref.loc12(%.loc12_28.2, %.loc12_43) [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_47: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_17.1: i32 = value_of_initializer %.loc12_21.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_21.3: i32 = converted %.loc12_21.2, %.loc12_17.1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_17.2: init i32 = call %Div.ref.loc12(%.loc12_21.3, %.loc12_47) [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_49: i32 = value_of_initializer %.loc12_17.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_17.3: i32 = converted %.loc12_17.2, %.loc12_49 [template = constants.%.5]
// CHECK:STDOUT:   %b.loc12: i32 = bind_name b, %.loc12_17.3
// CHECK:STDOUT:   %Div.ref.loc19: <function> = name_ref Div, %Div [template = %Div]
// CHECK:STDOUT:   %Sub.ref.loc19: <function> = name_ref Sub, %Sub [template = %Sub]
// CHECK:STDOUT:   %Negate.ref.loc19_22: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc19_29: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc19_28.1: init i32 = call %Negate.ref.loc19_22(%.loc19_29) [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_43: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc19_21.1: i32 = value_of_initializer %.loc19_28.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_28.2: i32 = converted %.loc19_28.1, %.loc19_21.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc19_21.2: init i32 = call %Sub.ref.loc19(%.loc19_28.2, %.loc19_43) [template = constants.%.5]
// CHECK:STDOUT:   %Negate.ref.loc19_47: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc19_54: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc19_53.1: init i32 = call %Negate.ref.loc19_47(%.loc19_54) [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_17.1: i32 = value_of_initializer %.loc19_21.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc19_21.3: i32 = converted %.loc19_21.2, %.loc19_17.1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc19_17.2: i32 = value_of_initializer %.loc19_53.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_53.2: i32 = converted %.loc19_53.1, %.loc19_17.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19_17.3: init i32 = call %Div.ref.loc19(%.loc19_21.3, %.loc19_53.2) [template = constants.%.5]
// CHECK:STDOUT:   %.loc19_57: i32 = value_of_initializer %.loc19_17.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc19_17.4: i32 = converted %.loc19_17.3, %.loc19_57 [template = constants.%.5]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc19_17.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a: i32, %b: i32) -> i32 = "int.div";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a: i32, %b: i32) -> i32 = "int.sub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_div_by_zero.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Div = %Div
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Div: <function> = fn_decl @Div [template] {
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Div.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Div.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Div.ref.loc10: <function> = name_ref Div, %Div [template = %Div]
// CHECK:STDOUT:   %.loc10_18: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_21: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_17.1: init i32 = call %Div.ref.loc10(%.loc10_18, %.loc10_21) [template = <error>]
// CHECK:STDOUT:   %.loc10_23: i32 = value_of_initializer %.loc10_17.1 [template = <error>]
// CHECK:STDOUT:   %.loc10_17.2: i32 = converted %.loc10_17.1, %.loc10_23 [template = <error>]
// CHECK:STDOUT:   %a.loc10: i32 = bind_name a, %.loc10_17.2
// CHECK:STDOUT:   %Div.ref.loc15: <function> = name_ref Div, %Div [template = %Div]
// CHECK:STDOUT:   %.loc15_18: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_17.1: init i32 = call %Div.ref.loc15(%.loc15_18, %.loc15_21) [template = <error>]
// CHECK:STDOUT:   %.loc15_23: i32 = value_of_initializer %.loc15_17.1 [template = <error>]
// CHECK:STDOUT:   %.loc15_17.2: i32 = converted %.loc15_17.1, %.loc15_23 [template = <error>]
// CHECK:STDOUT:   %b.loc15: i32 = bind_name b, %.loc15_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a: i32, %b: i32) -> i32 = "int.div";
// CHECK:STDOUT:
