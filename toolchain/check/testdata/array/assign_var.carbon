// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_var.carbon

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: --- assign_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.30, %.32, %.34) [template]
// CHECK:STDOUT:   %.35: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 0 [template]
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
// CHECK:STDOUT:   %int.make_type_32.loc11_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_22.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc11_9, %int.make_type_32.loc11_14, %int.make_type_32.loc11_19)
// CHECK:STDOUT:   %.loc11_22.2: type = value_of_initializer %int.make_type_32.loc11_9 [template = i32]
// CHECK:STDOUT:   %.loc11_22.3: type = converted %int.make_type_32.loc11_9, %.loc11_22.2 [template = i32]
// CHECK:STDOUT:   %.loc11_22.4: type = value_of_initializer %int.make_type_32.loc11_14 [template = i32]
// CHECK:STDOUT:   %.loc11_22.5: type = converted %int.make_type_32.loc11_14, %.loc11_22.4 [template = i32]
// CHECK:STDOUT:   %.loc11_22.6: type = value_of_initializer %int.make_type_32.loc11_19 [template = i32]
// CHECK:STDOUT:   %.loc11_22.7: type = converted %int.make_type_32.loc11_19, %.loc11_22.6 [template = i32]
// CHECK:STDOUT:   %.loc11_22.8: type = converted %.loc11_22.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_14: Core.IntLiteral = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_9.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_9.2: type = converted %int.make_type_32.loc12, %.loc12_9.1 [template = i32]
// CHECK:STDOUT:   %.loc12_15: type = array_type %.loc12_14, i32 [template = constants.%.35]
// CHECK:STDOUT:   %b.var: ref %.35 = var b
// CHECK:STDOUT:   %b: ref %.35 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_27: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_30: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_33: Core.IntLiteral = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_34.1: %tuple.type.3 = tuple_literal (%.loc11_27, %.loc11_30, %.loc11_33)
// CHECK:STDOUT:   %.loc11_34.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_34.3: <bound method> = bound_method %.loc11_27, %.loc11_34.2 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.1: init i32 = call %.loc11_34.3(%.loc11_27) [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_34.4: init i32 = converted %.loc11_27, %int.convert_checked.loc11_34.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_34.5: ref i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_34.6: init i32 = initialize_from %.loc11_34.4 to %.loc11_34.5 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_34.7: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_34.8: <bound method> = bound_method %.loc11_30, %.loc11_34.7 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.2: init i32 = call %.loc11_34.8(%.loc11_30) [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_34.9: init i32 = converted %.loc11_30, %int.convert_checked.loc11_34.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_34.10: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_34.11: init i32 = initialize_from %.loc11_34.9 to %.loc11_34.10 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_34.12: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_34.13: <bound method> = bound_method %.loc11_33, %.loc11_34.12 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.3: init i32 = call %.loc11_34.13(%.loc11_33) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_34.14: init i32 = converted %.loc11_33, %int.convert_checked.loc11_34.3 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_34.15: ref i32 = tuple_access file.%a.var, element2
// CHECK:STDOUT:   %.loc11_34.16: init i32 = initialize_from %.loc11_34.14 to %.loc11_34.15 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_34.17: init %tuple.type.2 = tuple_init (%.loc11_34.6, %.loc11_34.11, %.loc11_34.16) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_35: init %tuple.type.2 = converted %.loc11_34.1, %.loc11_34.17 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_35
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_19.1: ref i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_19.2: i32 = bind_value %.loc12_19.1
// CHECK:STDOUT:   %.loc12_19.3: i32 = int_value 0 [template = constants.%.37]
// CHECK:STDOUT:   %.loc12_19.4: ref i32 = array_index file.%b.var, %.loc12_19.3
// CHECK:STDOUT:   %.loc12_19.5: init i32 = initialize_from %.loc12_19.2 to %.loc12_19.4
// CHECK:STDOUT:   %.loc12_19.6: ref i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc12_19.7: i32 = bind_value %.loc12_19.6
// CHECK:STDOUT:   %.loc12_19.8: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_19.9: ref i32 = array_index file.%b.var, %.loc12_19.8
// CHECK:STDOUT:   %.loc12_19.10: init i32 = initialize_from %.loc12_19.7 to %.loc12_19.9
// CHECK:STDOUT:   %.loc12_19.11: ref i32 = tuple_access %a.ref, element2
// CHECK:STDOUT:   %.loc12_19.12: i32 = bind_value %.loc12_19.11
// CHECK:STDOUT:   %.loc12_19.13: i32 = int_value 2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc12_19.14: ref i32 = array_index file.%b.var, %.loc12_19.13
// CHECK:STDOUT:   %.loc12_19.15: init i32 = initialize_from %.loc12_19.12 to %.loc12_19.14
// CHECK:STDOUT:   %.loc12_19.16: init %.35 = array_init (%.loc12_19.5, %.loc12_19.10, %.loc12_19.15) to file.%b.var
// CHECK:STDOUT:   %.loc12_20: init %.35 = converted %a.ref, %.loc12_19.16
// CHECK:STDOUT:   assign file.%b.var, %.loc12_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
