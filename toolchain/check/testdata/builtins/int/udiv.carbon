// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/udiv.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/builtins/int/udiv.carbon

// --- int_div.carbon

fn Div(a: i32, b: i32) -> i32 = "int.udiv";

var arr: [i32; Div(3, 2)];
let arr_p: [i32; 1]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Div(a, b);
}

// --- overflow.carbon

package Overflow;

fn Div(a: i32, b: i32) -> i32 = "int.udiv";
fn Sub(a: i32, b: i32) -> i32 = "int.usub";
fn Negate(a: i32) -> i32 = "int.unegate";

// -0x7FFF_FFFF / -1 is OK.
let a: i32 = Div(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 / 1 is OK.
let b: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 is OK.
let c: i32 = Div(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

package FailDivByZero;

fn Div(a: i32, b: i32) -> i32 = "int.udiv";

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
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Div: type = fn_type @Div [template]
// CHECK:STDOUT:   %struct.2: Div = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %RuntimeCall: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %struct.3: RuntimeCall = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Div.decl: Div = fn_decl @Div [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc2_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_32.loc2_11 [template = i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_32.loc2_11, %.loc2_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Div.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %int.make_type_32.loc2_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_32.loc2_19 [template = i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_32.loc2_19, %.loc2_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Div.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %int.make_type_32.loc2_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_32.loc2_27 [template = i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_32.loc2_27, %.loc2_27.1 [template = i32]
// CHECK:STDOUT:     @Div.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %Div.ref: Div = name_ref Div, %Div.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc4_20: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_23: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.udiv: init i32 = call %Div.ref(%.loc4_20, %.loc4_23) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_25: type = array_type %int.udiv, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref [i32; 1] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 1] = bind_name arr, %arr.var
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %.loc5_18, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type [i32; 1] [template = constants.%.6]
// CHECK:STDOUT:   %arr.ref: ref [i32; 1] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_24: [i32; 1]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 1]* = bind_name arr_p, %.loc5_24
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %RuntimeCall.decl: RuntimeCall = fn_decl @RuntimeCall [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc7_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_32.loc7_19 [template = i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_32.loc7_19, %.loc7_19.1 [template = i32]
// CHECK:STDOUT:     %a.loc7_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %int.make_type_32.loc7_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_32.loc7_27 [template = i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_32.loc7_27, %.loc7_27.1 [template = i32]
// CHECK:STDOUT:     %b.loc7_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %int.make_type_32.loc7_35: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_32.loc7_35 [template = i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_32.loc7_35, %.loc7_35.1 [template = i32]
// CHECK:STDOUT:     @RuntimeCall.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a: i32, %b: i32) -> i32 = "int.udiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Div.ref: Div = name_ref Div, file.%Div.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.udiv: init i32 = call %Div.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: i32 = value_of_initializer %int.udiv
// CHECK:STDOUT:   %.loc8_19.2: i32 = converted %int.udiv, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Div: type = fn_type @Div [template]
// CHECK:STDOUT:   %struct.2: Div = struct_value () [template]
// CHECK:STDOUT:   %Sub: type = fn_type @Sub [template]
// CHECK:STDOUT:   %struct.3: Sub = struct_value () [template]
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %struct.4: Negate = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal -2147483647 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal -1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal -2147483648 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Div.decl: Div = fn_decl @Div [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc4_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4_11 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4_11, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Div.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %int.make_type_32.loc4_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %int.make_type_32.loc4_19 [template = i32]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %int.make_type_32.loc4_19, %.loc4_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Div.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     %int.make_type_32.loc4_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4_27 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4_27, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     @Div.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Sub.decl: Sub = fn_decl @Sub [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc5_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc5_11.1: type = value_of_initializer %int.make_type_32.loc5_11 [template = i32]
// CHECK:STDOUT:     %.loc5_11.2: type = converted %int.make_type_32.loc5_11, %.loc5_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc5_8.1: i32 = param a
// CHECK:STDOUT:     @Sub.%a: i32 = bind_name a, %a.loc5_8.1
// CHECK:STDOUT:     %int.make_type_32.loc5_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc5_19.1: type = value_of_initializer %int.make_type_32.loc5_19 [template = i32]
// CHECK:STDOUT:     %.loc5_19.2: type = converted %int.make_type_32.loc5_19, %.loc5_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc5_16.1: i32 = param b
// CHECK:STDOUT:     @Sub.%b: i32 = bind_name b, %b.loc5_16.1
// CHECK:STDOUT:     %int.make_type_32.loc5_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc5_27.1: type = value_of_initializer %int.make_type_32.loc5_27 [template = i32]
// CHECK:STDOUT:     %.loc5_27.2: type = converted %int.make_type_32.loc5_27, %.loc5_27.1 [template = i32]
// CHECK:STDOUT:     @Sub.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct.4] {
// CHECK:STDOUT:     %int.make_type_32.loc6_14: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc6_14.1: type = value_of_initializer %int.make_type_32.loc6_14 [template = i32]
// CHECK:STDOUT:     %.loc6_14.2: type = converted %int.make_type_32.loc6_14, %.loc6_14.1 [template = i32]
// CHECK:STDOUT:     %a.loc6_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc6_11.1
// CHECK:STDOUT:     %int.make_type_32.loc6_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc6_22.1: type = value_of_initializer %int.make_type_32.loc6_22 [template = i32]
// CHECK:STDOUT:     %.loc6_22.2: type = converted %int.make_type_32.loc6_22, %.loc6_22.1 [template = i32]
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc9: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_8.1: type = value_of_initializer %int.make_type_32.loc9 [template = i32]
// CHECK:STDOUT:   %.loc9_8.2: type = converted %int.make_type_32.loc9, %.loc9_8.1 [template = i32]
// CHECK:STDOUT:   %Div.ref.loc9: Div = name_ref Div, %Div.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %Negate.ref.loc9_18: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc9_25: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.unegate.loc9_24: init i32 = call %Negate.ref.loc9_18(%.loc9_25) [template = constants.%.3]
// CHECK:STDOUT:   %Negate.ref.loc9_39: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc9_46: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.unegate.loc9_45: init i32 = call %Negate.ref.loc9_39(%.loc9_46) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_17.1: i32 = value_of_initializer %int.unegate.loc9_24 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_17.2: i32 = converted %int.unegate.loc9_24, %.loc9_17.1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_17.3: i32 = value_of_initializer %int.unegate.loc9_45 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_17.4: i32 = converted %int.unegate.loc9_45, %.loc9_17.3 [template = constants.%.5]
// CHECK:STDOUT:   %int.udiv.loc9: init i32 = call %Div.ref.loc9(%.loc9_17.2, %.loc9_17.4) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_49.1: i32 = value_of_initializer %int.udiv.loc9 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_49.2: i32 = converted %int.udiv.loc9, %.loc9_49.1 [template = constants.%.6]
// CHECK:STDOUT:   %a.loc9: i32 = bind_name a, %.loc9_49.2
// CHECK:STDOUT:   %import_ref.10: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_32.loc12, %.loc12_8.1 [template = i32]
// CHECK:STDOUT:   %Div.ref.loc12: Div = name_ref Div, %Div.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %Sub.ref.loc12: Sub = name_ref Sub, %Sub.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %Negate.ref.loc12: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc12_29: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.unegate.loc12: init i32 = call %Negate.ref.loc12(%.loc12_29) [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_43: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_21.1: i32 = value_of_initializer %int.unegate.loc12 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_21.2: i32 = converted %int.unegate.loc12, %.loc12_21.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.usub.loc12: init i32 = call %Sub.ref.loc12(%.loc12_21.2, %.loc12_43) [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_47: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_17.1: i32 = value_of_initializer %int.usub.loc12 [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_17.2: i32 = converted %int.usub.loc12, %.loc12_17.1 [template = constants.%.7]
// CHECK:STDOUT:   %int.udiv.loc12: init i32 = call %Div.ref.loc12(%.loc12_17.2, %.loc12_47) [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_49.1: i32 = value_of_initializer %int.udiv.loc12 [template = constants.%.7]
// CHECK:STDOUT:   %.loc12_49.2: i32 = converted %int.udiv.loc12, %.loc12_49.1 [template = constants.%.7]
// CHECK:STDOUT:   %b.loc12: i32 = bind_name b, %.loc12_49.2
// CHECK:STDOUT:   %import_ref.11: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %Div.ref.loc15: Div = name_ref Div, %Div.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %Sub.ref.loc15: Sub = name_ref Sub, %Sub.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %Negate.ref.loc15_22: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc15_29: i32 = int_literal 2147483647 [template = constants.%.2]
// CHECK:STDOUT:   %int.unegate.loc15_28: init i32 = call %Negate.ref.loc15_22(%.loc15_29) [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_43: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15_21.1: i32 = value_of_initializer %int.unegate.loc15_28 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_21.2: i32 = converted %int.unegate.loc15_28, %.loc15_21.1 [template = constants.%.3]
// CHECK:STDOUT:   %int.usub.loc15: init i32 = call %Sub.ref.loc15(%.loc15_21.2, %.loc15_43) [template = constants.%.7]
// CHECK:STDOUT:   %Negate.ref.loc15_47: Negate = name_ref Negate, %Negate.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc15_54: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %int.unegate.loc15_53: init i32 = call %Negate.ref.loc15_47(%.loc15_54) [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_17.1: i32 = value_of_initializer %int.usub.loc15 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_17.2: i32 = converted %int.usub.loc15, %.loc15_17.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_17.3: i32 = value_of_initializer %int.unegate.loc15_53 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_17.4: i32 = converted %int.unegate.loc15_53, %.loc15_17.3 [template = constants.%.5]
// CHECK:STDOUT:   %int.udiv.loc15: init i32 = call %Div.ref.loc15(%.loc15_17.2, %.loc15_17.4) [template = constants.%.6]
// CHECK:STDOUT:   %.loc15_57.1: i32 = value_of_initializer %int.udiv.loc15 [template = constants.%.6]
// CHECK:STDOUT:   %.loc15_57.2: i32 = converted %int.udiv.loc15, %.loc15_57.1 [template = constants.%.6]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc15_57.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a: i32, %b: i32) -> i32 = "int.udiv";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a: i32, %b: i32) -> i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.unegate";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_div_by_zero.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Div: type = fn_type @Div [template]
// CHECK:STDOUT:   %struct.2: Div = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Div = %Div.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Div.decl: Div = fn_decl @Div [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc4_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4_11 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4_11, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %a.loc4_8.1: i32 = param a
// CHECK:STDOUT:     @Div.%a: i32 = bind_name a, %a.loc4_8.1
// CHECK:STDOUT:     %int.make_type_32.loc4_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %int.make_type_32.loc4_19 [template = i32]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %int.make_type_32.loc4_19, %.loc4_19.1 [template = i32]
// CHECK:STDOUT:     %b.loc4_16.1: i32 = param b
// CHECK:STDOUT:     @Div.%b: i32 = bind_name b, %b.loc4_16.1
// CHECK:STDOUT:     %int.make_type_32.loc4_27: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_32.loc4_27 [template = i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_32.loc4_27, %.loc4_27.1 [template = i32]
// CHECK:STDOUT:     @Div.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc10: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc10_8.1: type = value_of_initializer %int.make_type_32.loc10 [template = i32]
// CHECK:STDOUT:   %.loc10_8.2: type = converted %int.make_type_32.loc10, %.loc10_8.1 [template = i32]
// CHECK:STDOUT:   %Div.ref.loc10: Div = name_ref Div, %Div.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc10_18: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_21: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.udiv.loc10: init i32 = call %Div.ref.loc10(%.loc10_18, %.loc10_21) [template = <error>]
// CHECK:STDOUT:   %.loc10_23.1: i32 = value_of_initializer %int.udiv.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_23.2: i32 = converted %int.udiv.loc10, %.loc10_23.1 [template = <error>]
// CHECK:STDOUT:   %a.loc10: i32 = bind_name a, %.loc10_23.2
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %Div.ref.loc15: Div = name_ref Div, %Div.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc15_18: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %int.udiv.loc15: init i32 = call %Div.ref.loc15(%.loc15_18, %.loc15_21) [template = <error>]
// CHECK:STDOUT:   %.loc15_23.1: i32 = value_of_initializer %int.udiv.loc15 [template = <error>]
// CHECK:STDOUT:   %.loc15_23.2: i32 = converted %int.udiv.loc15, %.loc15_23.1 [template = <error>]
// CHECK:STDOUT:   %b.loc15: i32 = bind_name b, %.loc15_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Div(%a: i32, %b: i32) -> i32 = "int.udiv";
// CHECK:STDOUT:
