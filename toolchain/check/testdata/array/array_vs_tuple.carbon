// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon

fn G() {
  // These should have two different constant values.
  var a: [i32; 3] = (1, 2, 3);
  var b: (i32, i32, i32) = (1, 2, 3);
}

// CHECK:STDOUT: --- array_vs_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.6: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %.2 = tuple_value (%.32, %.34, %.36) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.3 = tuple_value (%.32, %.34, %.36) [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_16: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_11.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_11.2: type = converted %int.make_type_32.loc13, %.loc13_11.1 [template = i32]
// CHECK:STDOUT:   %.loc13_17: type = array_type %.loc13_16, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_22: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_25: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_28: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_29.1: %tuple.type.1 = tuple_literal (%.loc13_22, %.loc13_25, %.loc13_28)
// CHECK:STDOUT:   %.loc13_29.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_29.3: <bound method> = bound_method %.loc13_22, %.loc13_29.2 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.1: init i32 = call %.loc13_29.3(%.loc13_22) [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_29.4: init i32 = converted %.loc13_22, %int.convert_checked.loc13_29.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_29.5: i32 = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_29.6: ref i32 = array_index %a.var, %.loc13_29.5
// CHECK:STDOUT:   %.loc13_29.7: init i32 = initialize_from %.loc13_29.4 to %.loc13_29.6 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_29.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_29.9: <bound method> = bound_method %.loc13_25, %.loc13_29.8 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.2: init i32 = call %.loc13_29.9(%.loc13_25) [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_29.10: init i32 = converted %.loc13_25, %int.convert_checked.loc13_29.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_29.11: i32 = int_value 1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_29.12: ref i32 = array_index %a.var, %.loc13_29.11
// CHECK:STDOUT:   %.loc13_29.13: init i32 = initialize_from %.loc13_29.10 to %.loc13_29.12 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_29.14: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_29.15: <bound method> = bound_method %.loc13_28, %.loc13_29.14 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.3: init i32 = call %.loc13_29.15(%.loc13_28) [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_29.16: init i32 = converted %.loc13_28, %int.convert_checked.loc13_29.3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_29.17: i32 = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_29.18: ref i32 = array_index %a.var, %.loc13_29.17
// CHECK:STDOUT:   %.loc13_29.19: init i32 = initialize_from %.loc13_29.16 to %.loc13_29.18 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_29.20: init %.2 = array_init (%.loc13_29.7, %.loc13_29.13, %.loc13_29.19) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_30: init %.2 = converted %.loc13_29.1, %.loc13_29.20 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_30
// CHECK:STDOUT:   %int.make_type_32.loc14_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_24.1: %tuple.type.2 = tuple_literal (%int.make_type_32.loc14_11, %int.make_type_32.loc14_16, %int.make_type_32.loc14_21)
// CHECK:STDOUT:   %.loc14_24.2: type = value_of_initializer %int.make_type_32.loc14_11 [template = i32]
// CHECK:STDOUT:   %.loc14_24.3: type = converted %int.make_type_32.loc14_11, %.loc14_24.2 [template = i32]
// CHECK:STDOUT:   %.loc14_24.4: type = value_of_initializer %int.make_type_32.loc14_16 [template = i32]
// CHECK:STDOUT:   %.loc14_24.5: type = converted %int.make_type_32.loc14_16, %.loc14_24.4 [template = i32]
// CHECK:STDOUT:   %.loc14_24.6: type = value_of_initializer %int.make_type_32.loc14_21 [template = i32]
// CHECK:STDOUT:   %.loc14_24.7: type = converted %int.make_type_32.loc14_21, %.loc14_24.6 [template = i32]
// CHECK:STDOUT:   %.loc14_24.8: type = converted %.loc14_24.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:   %b.var: ref %tuple.type.3 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.3 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14_29: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_32: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_35: Core.IntLiteral = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_36.1: %tuple.type.1 = tuple_literal (%.loc14_29, %.loc14_32, %.loc14_35)
// CHECK:STDOUT:   %.loc14_36.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_36.3: <bound method> = bound_method %.loc14_29, %.loc14_36.2 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.1: init i32 = call %.loc14_36.3(%.loc14_29) [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_36.4: init i32 = converted %.loc14_29, %int.convert_checked.loc14_36.1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_36.5: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc14_36.6: init i32 = initialize_from %.loc14_36.4 to %.loc14_36.5 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_36.7: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_36.8: <bound method> = bound_method %.loc14_32, %.loc14_36.7 [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.2: init i32 = call %.loc14_36.8(%.loc14_32) [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_36.9: init i32 = converted %.loc14_32, %int.convert_checked.loc14_36.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_36.10: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc14_36.11: init i32 = initialize_from %.loc14_36.9 to %.loc14_36.10 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_36.12: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_36.13: <bound method> = bound_method %.loc14_35, %.loc14_36.12 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.3: init i32 = call %.loc14_36.13(%.loc14_35) [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_36.14: init i32 = converted %.loc14_35, %int.convert_checked.loc14_36.3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_36.15: ref i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc14_36.16: init i32 = initialize_from %.loc14_36.14 to %.loc14_36.15 [template = constants.%.36]
// CHECK:STDOUT:   %.loc14_36.17: init %tuple.type.3 = tuple_init (%.loc14_36.6, %.loc14_36.11, %.loc14_36.16) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_37: init %tuple.type.3 = converted %.loc14_36.1, %.loc14_36.17 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc14_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
