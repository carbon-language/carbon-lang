// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_out_of_bound_non_literal.carbon

var a: [i32; 3] = (1, 2, 3);
// CHECK:STDERR: fail_out_of_bound_non_literal.carbon:[[@LINE+3]]:16: error: array index `3` is past the end of type `[i32; 3]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a[{.index = 3}.index];
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
var b: i32 = a[{.index = 3}.index];

// CHECK:STDOUT: --- fail_out_of_bound_non_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.28: type = array_type %.27, i32 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.32: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.28 = tuple_value (%.30, %.31, %.1) [template]
// CHECK:STDOUT:   %.33: type = struct_type {.index: i32} [template]
// CHECK:STDOUT:   %struct: %.33 = struct_value (%.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32.loc11, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_14.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_14.3: <bound method> = bound_method %.loc11_14.1, %.loc11_14.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc11_14.3(%.loc11_14.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_14.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_14.5: Core.IntLiteral = converted %.loc11_14.1, %.loc11_14.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14.5, i32 [template = constants.%.28]
// CHECK:STDOUT:   %a.var: ref %.28 = var a
// CHECK:STDOUT:   %a: ref %.28 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_32.loc15, %.loc15_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_23: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_26: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_27.3: ref i32 = array_index file.%a.var, %.loc11_27.2
// CHECK:STDOUT:   %.loc11_27.4: init i32 = initialize_from %.loc11_20 to %.loc11_27.3 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_27.5: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_27.6: ref i32 = array_index file.%a.var, %.loc11_27.5
// CHECK:STDOUT:   %.loc11_27.7: init i32 = initialize_from %.loc11_23 to %.loc11_27.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_27.8: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_27.9: ref i32 = array_index file.%a.var, %.loc11_27.8
// CHECK:STDOUT:   %.loc11_27.10: init i32 = initialize_from %.loc11_26 to %.loc11_27.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_27.11: init %.28 = array_init (%.loc11_27.4, %.loc11_27.7, %.loc11_27.10) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_28: init %.28 = converted %.loc11_27.1, %.loc11_27.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %.28 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc15_26: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc15_27.1: %.33 = struct_literal (%.loc15_26)
// CHECK:STDOUT:   %struct: %.33 = struct_value (%.loc15_26) [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_27.2: %.33 = converted %.loc15_27.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_28: i32 = struct_access %.loc15_27.2, element0 [template = constants.%.1]
// CHECK:STDOUT:   %.loc15_34.1: ref i32 = array_index %a.ref, %.loc15_28 [template = <error>]
// CHECK:STDOUT:   %.loc15_34.2: i32 = bind_value %.loc15_34.1
// CHECK:STDOUT:   assign file.%b.var, %.loc15_34.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
