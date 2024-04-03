// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_div.carbon

fn Mod(a: i32, b: i32) -> i32 = "int.mod";

var arr: [i32; Mod(5, 3)];
let arr_p: [i32; 2]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Mod(a, b);
}

// --- fail_overflow.carbon

package FailOverflow api;

fn Mod(a: i32, b: i32) -> i32 = "int.mod";
fn Sub(a: i32, b: i32) -> i32 = "int.sub";
fn Negate(a: i32) -> i32 = "int.negate";

// -0x7FFF_FFFF % -1 is OK.
let a: i32 = Mod(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 % 1 is OK.
let b: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 overflows, so -0x8000_0000 % -1 is disallowed, even though
// its result is representable.
// CHECK:STDERR: fail_overflow.carbon:[[@LINE+4]]:14: ERROR: Integer overflow in calculation -2147483648 % -1.
// CHECK:STDERR: let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

package FailDivByZero api;

fn Mod(a: i32, b: i32) -> i32 = "int.mod";

// Remainder of division by zero is not defined.

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+4]]:14: ERROR: Division by zero.
// CHECK:STDERR: let a: i32 = Mod(1, 0);
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
let a: i32 = Mod(1, 0);

// CHECK:STDERR: fail_div_by_zero.carbon:[[@LINE+3]]:14: ERROR: Division by zero.
// CHECK:STDERR: let b: i32 = Mod(0, 0);
// CHECK:STDERR:              ^~~~
let b: i32 = Mod(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: type = array_type %.3, i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type [i32; 2] [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Mod = %Mod
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Mod: <function> = fn_decl @Mod [template] {
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Mod.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Mod.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %return.var.loc2: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Mod.ref: <function> = name_ref Mod, %Mod [template = %Mod]
// CHECK:STDOUT:   %.loc4_20: i32 = int_literal 5 [template = constants.%.1]
// CHECK:STDOUT:   %.loc4_23: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_19: init i32 = call %Mod.ref(%.loc4_20, %.loc4_23) [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_19, i32 [template = constants.%.4]
// CHECK:STDOUT:   %arr.var: ref [i32; 2] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 2] = bind_name arr, %arr.var
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type [i32; 2] [template = constants.%.5]
// CHECK:STDOUT:   %arr.ref: ref [i32; 2] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_24: [i32; 2]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 2]* = bind_name arr_p, %.loc5_24
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod(%a: i32, %b: i32) -> i32 = "int.mod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref: <function> = name_ref Mod, file.%Mod [template = file.%Mod]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc8_13.1: init i32 = call %Mod.ref(%a.ref, %b.ref)
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
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Mod = %Mod
// CHECK:STDOUT:     .Sub = %Sub
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Mod: <function> = fn_decl @Mod [template] {
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Mod.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Mod.%b: i32 = bind_name b, %b.loc4_16.1
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
// CHECK:STDOUT:   %Mod.ref.loc9: <function> = name_ref Mod, %Mod [template = %Mod]
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
// CHECK:STDOUT:   %.loc9_17.3: init i32 = call %Mod.ref.loc9(%.loc9_24.2, %.loc9_45.2) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_49: i32 = value_of_initializer %.loc9_17.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_17.4: i32 = converted %.loc9_17.3, %.loc9_49 [template = constants.%.5]
// CHECK:STDOUT:   %a.loc9: i32 = bind_name a, %.loc9_17.4
// CHECK:STDOUT:   %Mod.ref.loc12: <function> = name_ref Mod, %Mod [template = %Mod]
// CHECK:STDOUT:   %Sub.ref.loc12: <function> = name_ref Sub, %Sub [template = %Sub]
// CHECK:STDOUT:   %Negate.ref.loc12: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc12_29: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_28.1: init i32 = call %Negate.ref.loc12(%.loc12_29) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_43: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_21.1: i32 = value_of_initializer %.loc12_28.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_28.2: i32 = converted %.loc12_28.1, %.loc12_21.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_21.2: init i32 = call %Sub.ref.loc12(%.loc12_28.2, %.loc12_43) [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_47: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_17.1: i32 = value_of_initializer %.loc12_21.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_21.3: i32 = converted %.loc12_21.2, %.loc12_17.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_17.2: init i32 = call %Mod.ref.loc12(%.loc12_21.3, %.loc12_47) [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_49: i32 = value_of_initializer %.loc12_17.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_17.3: i32 = converted %.loc12_17.2, %.loc12_49 [template = constants.%.5]
// CHECK:STDOUT:   %b.loc12: i32 = bind_name b, %.loc12_17.3
// CHECK:STDOUT:   %Mod.ref.loc20: <function> = name_ref Mod, %Mod [template = %Mod]
// CHECK:STDOUT:   %Sub.ref.loc20: <function> = name_ref Sub, %Sub [template = %Sub]
// CHECK:STDOUT:   %Negate.ref.loc20_22: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc20_29: i32 = int_literal 2147483647 [template = constants.%.1]
// CHECK:STDOUT:   %.loc20_28.1: init i32 = call %Negate.ref.loc20_22(%.loc20_29) [template = constants.%.2]
// CHECK:STDOUT:   %.loc20_43: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc20_21.1: i32 = value_of_initializer %.loc20_28.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc20_28.2: i32 = converted %.loc20_28.1, %.loc20_21.1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc20_21.2: init i32 = call %Sub.ref.loc20(%.loc20_28.2, %.loc20_43) [template = constants.%.6]
// CHECK:STDOUT:   %Negate.ref.loc20_47: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc20_54: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc20_53.1: init i32 = call %Negate.ref.loc20_47(%.loc20_54) [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_17.1: i32 = value_of_initializer %.loc20_21.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc20_21.3: i32 = converted %.loc20_21.2, %.loc20_17.1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc20_17.2: i32 = value_of_initializer %.loc20_53.1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_53.2: i32 = converted %.loc20_53.1, %.loc20_17.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_17.3: init i32 = call %Mod.ref.loc20(%.loc20_21.3, %.loc20_53.2) [template = constants.%.5]
// CHECK:STDOUT:   %.loc20_57: i32 = value_of_initializer %.loc20_17.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc20_17.4: i32 = converted %.loc20_17.3, %.loc20_57 [template = constants.%.5]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc20_17.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod(%a: i32, %b: i32) -> i32 = "int.mod";
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
// CHECK:STDOUT:     .Mod = %Mod
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Mod: <function> = fn_decl @Mod [template] {
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Mod.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Mod.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Mod.ref.loc12: <function> = name_ref Mod, %Mod [template = %Mod]
// CHECK:STDOUT:   %.loc12_18: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_21: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_17.1: init i32 = call %Mod.ref.loc12(%.loc12_18, %.loc12_21) [template = <error>]
// CHECK:STDOUT:   %.loc12_23: i32 = value_of_initializer %.loc12_17.1 [template = <error>]
// CHECK:STDOUT:   %.loc12_17.2: i32 = converted %.loc12_17.1, %.loc12_23 [template = <error>]
// CHECK:STDOUT:   %a.loc12: i32 = bind_name a, %.loc12_17.2
// CHECK:STDOUT:   %Mod.ref.loc17: <function> = name_ref Mod, %Mod [template = %Mod]
// CHECK:STDOUT:   %.loc17_18: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_21: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_17.1: init i32 = call %Mod.ref.loc17(%.loc17_18, %.loc17_21) [template = <error>]
// CHECK:STDOUT:   %.loc17_23: i32 = value_of_initializer %.loc17_17.1 [template = <error>]
// CHECK:STDOUT:   %.loc17_17.2: i32 = converted %.loc17_17.1, %.loc17_23 [template = <error>]
// CHECK:STDOUT:   %b.loc17: i32 = bind_name b, %.loc17_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod(%a: i32, %b: i32) -> i32 = "int.mod";
// CHECK:STDOUT:
