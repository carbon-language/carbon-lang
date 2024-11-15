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
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.32: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.28 = tuple_value (%.30, %.31, %.1) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.30, %.31, %.1) [template]
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
// CHECK:STDOUT:   %.loc13_16.1: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_11.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_11.2: type = converted %int.make_type_32.loc13, %.loc13_11.1 [template = i32]
// CHECK:STDOUT:   %.loc13_16.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_16.3: <bound method> = bound_method %.loc13_16.1, %.loc13_16.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc13_16.3(%.loc13_16.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc13_16.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc13_16.5: Core.IntLiteral = converted %.loc13_16.1, %.loc13_16.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc13_17: type = array_type %.loc13_16.5, i32 [template = constants.%.28]
// CHECK:STDOUT:   %a.var: ref %.28 = var a
// CHECK:STDOUT:   %a: ref %.28 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_22: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_25: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_28: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_29.1: %tuple.type.1 = tuple_literal (%.loc13_22, %.loc13_25, %.loc13_28)
// CHECK:STDOUT:   %.loc13_29.2: i32 = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_29.3: ref i32 = array_index %a.var, %.loc13_29.2
// CHECK:STDOUT:   %.loc13_29.4: init i32 = initialize_from %.loc13_22 to %.loc13_29.3 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_29.5: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_29.6: ref i32 = array_index %a.var, %.loc13_29.5
// CHECK:STDOUT:   %.loc13_29.7: init i32 = initialize_from %.loc13_25 to %.loc13_29.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_29.8: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_29.9: ref i32 = array_index %a.var, %.loc13_29.8
// CHECK:STDOUT:   %.loc13_29.10: init i32 = initialize_from %.loc13_28 to %.loc13_29.9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc13_29.11: init %.28 = array_init (%.loc13_29.4, %.loc13_29.7, %.loc13_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_30: init %.28 = converted %.loc13_29.1, %.loc13_29.11 [template = constants.%array]
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
// CHECK:STDOUT:   %.loc14_24.8: type = converted %.loc14_24.1, constants.%tuple.type.1 [template = constants.%tuple.type.1]
// CHECK:STDOUT:   %b.var: ref %tuple.type.1 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.1 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14_29: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_32: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_35: i32 = int_value 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_36.1: %tuple.type.1 = tuple_literal (%.loc14_29, %.loc14_32, %.loc14_35)
// CHECK:STDOUT:   %.loc14_36.2: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc14_36.3: init i32 = initialize_from %.loc14_29 to %.loc14_36.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_36.4: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc14_36.5: init i32 = initialize_from %.loc14_32 to %.loc14_36.4 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_36.6: ref i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc14_36.7: init i32 = initialize_from %.loc14_35 to %.loc14_36.6 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_36.8: init %tuple.type.1 = tuple_init (%.loc14_36.3, %.loc14_36.5, %.loc14_36.7) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_37: init %tuple.type.1 = converted %.loc14_36.1, %.loc14_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc14_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
