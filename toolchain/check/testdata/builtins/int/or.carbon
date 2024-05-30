// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/builtins/int/or.carbon

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
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Or: type = fn_type @Or [template]
// CHECK:STDOUT:   %struct.2: Or = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 10 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 14 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type [i32; 14] [template]
// CHECK:STDOUT:   %RuntimeCall: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %struct.3: RuntimeCall = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Or = %Or.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Or.decl: Or = fn_decl @Or [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc2_10: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_10.1: type = value_of_initializer %int.make_type_32.loc2_10 [template = i32]
// CHECK:STDOUT:     %.loc2_10.2: type = converted %int.make_type_32.loc2_10, %.loc2_10.1 [template = i32]
// CHECK:STDOUT:     %a.loc2_7.1: i32 = param a
// CHECK:STDOUT:     @Or.%a: i32 = bind_name a, %a.loc2_7.1
// CHECK:STDOUT:     %int.make_type_32.loc2_18: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_18.1: type = value_of_initializer %int.make_type_32.loc2_18 [template = i32]
// CHECK:STDOUT:     %.loc2_18.2: type = converted %int.make_type_32.loc2_18, %.loc2_18.1 [template = i32]
// CHECK:STDOUT:     %b.loc2_15.1: i32 = param b
// CHECK:STDOUT:     @Or.%b: i32 = bind_name b, %b.loc2_15.1
// CHECK:STDOUT:     %int.make_type_32.loc2_26: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc2_26.1: type = value_of_initializer %int.make_type_32.loc2_26 [template = i32]
// CHECK:STDOUT:     %.loc2_26.2: type = converted %int.make_type_32.loc2_26, %.loc2_26.1 [template = i32]
// CHECK:STDOUT:     @Or.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %Or.ref: Or = name_ref Or, %Or.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc4_19: i32 = int_literal 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_23: i32 = int_literal 10 [template = constants.%.3]
// CHECK:STDOUT:   %int.or: init i32 = call %Or.ref(%.loc4_19, %.loc4_23) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_26: type = array_type %int.or, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref [i32; 14] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 14] = bind_name arr, %arr.var
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc5_18: i32 = int_literal 14 [template = constants.%.4]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_32.loc5, %.loc5_13.1 [template = i32]
// CHECK:STDOUT:   %.loc5_20: type = array_type %.loc5_18, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc5_21: type = ptr_type [i32; 14] [template = constants.%.6]
// CHECK:STDOUT:   %arr.ref: ref [i32; 14] = name_ref arr, %arr
// CHECK:STDOUT:   %.loc5_25: [i32; 14]* = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: [i32; 14]* = bind_name arr_p, %.loc5_25
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
// CHECK:STDOUT: fn @Or(%a: i32, %b: i32) -> i32 = "int.or";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Or.ref: Or = name_ref Or, file.%Or.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %int.or: init i32 = call %Or.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_18.1: i32 = value_of_initializer %int.or
// CHECK:STDOUT:   %.loc8_18.2: i32 = converted %int.or, %.loc8_18.1
// CHECK:STDOUT:   return %.loc8_18.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
