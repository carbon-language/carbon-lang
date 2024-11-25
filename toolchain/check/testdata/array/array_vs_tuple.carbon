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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.31: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.39: <specific function> = specific_function %.38, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.40: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.34, %.37, %.40) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.3 = tuple_value (%.34, %.37, %.40) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
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
// CHECK:STDOUT:   %.loc13_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_16: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_11.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_11.3: type = converted %int.make_type_signed.loc13, %.loc13_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_17: type = array_type %.loc13_16, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc13_22: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_25: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_28: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc13_29.1: %tuple.type.1 = tuple_literal (%.loc13_22, %.loc13_25, %.loc13_28)
// CHECK:STDOUT:   %.loc13_29.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_29.3: <bound method> = bound_method %.loc13_22, %.loc13_29.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_29.4: <specific function> = specific_function %.loc13_29.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.1: init %i32 = call %.loc13_29.4(%.loc13_22) [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_29.5: init %i32 = converted %.loc13_22, %int.convert_checked.loc13_29.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_29.6: Core.IntLiteral = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_29.7: ref %i32 = array_index %a.var, %.loc13_29.6
// CHECK:STDOUT:   %.loc13_29.8: init %i32 = initialize_from %.loc13_29.5 to %.loc13_29.7 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_29.9: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_29.10: <bound method> = bound_method %.loc13_25, %.loc13_29.9 [template = constants.%.35]
// CHECK:STDOUT:   %.loc13_29.11: <specific function> = specific_function %.loc13_29.10, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.2: init %i32 = call %.loc13_29.11(%.loc13_25) [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_29.12: init %i32 = converted %.loc13_25, %int.convert_checked.loc13_29.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_29.13: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_29.14: ref %i32 = array_index %a.var, %.loc13_29.13
// CHECK:STDOUT:   %.loc13_29.15: init %i32 = initialize_from %.loc13_29.12 to %.loc13_29.14 [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_29.16: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_29.17: <bound method> = bound_method %.loc13_28, %.loc13_29.16 [template = constants.%.38]
// CHECK:STDOUT:   %.loc13_29.18: <specific function> = specific_function %.loc13_29.17, @Convert.2(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.3: init %i32 = call %.loc13_29.18(%.loc13_28) [template = constants.%.40]
// CHECK:STDOUT:   %.loc13_29.19: init %i32 = converted %.loc13_28, %int.convert_checked.loc13_29.3 [template = constants.%.40]
// CHECK:STDOUT:   %.loc13_29.20: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_29.21: ref %i32 = array_index %a.var, %.loc13_29.20
// CHECK:STDOUT:   %.loc13_29.22: init %i32 = initialize_from %.loc13_29.19 to %.loc13_29.21 [template = constants.%.40]
// CHECK:STDOUT:   %.loc13_29.23: init %.3 = array_init (%.loc13_29.8, %.loc13_29.15, %.loc13_29.22) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_30: init %.3 = converted %.loc13_29.1, %.loc13_29.23 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_30
// CHECK:STDOUT:   %.loc14_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_11: init type = call constants.%Int(%.loc14_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_16: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_16: init type = call constants.%Int(%.loc14_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_21: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_21: init type = call constants.%Int(%.loc14_21) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.1: %tuple.type.2 = tuple_literal (%int.make_type_signed.loc14_11, %int.make_type_signed.loc14_16, %int.make_type_signed.loc14_21)
// CHECK:STDOUT:   %.loc14_24.2: type = value_of_initializer %int.make_type_signed.loc14_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.3: type = converted %int.make_type_signed.loc14_11, %.loc14_24.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.4: type = value_of_initializer %int.make_type_signed.loc14_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.5: type = converted %int.make_type_signed.loc14_16, %.loc14_24.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.6: type = value_of_initializer %int.make_type_signed.loc14_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.7: type = converted %int.make_type_signed.loc14_21, %.loc14_24.6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.8: type = converted %.loc14_24.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:   %b.var: ref %tuple.type.3 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.3 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14_29: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_32: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_35: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_36.1: %tuple.type.1 = tuple_literal (%.loc14_29, %.loc14_32, %.loc14_35)
// CHECK:STDOUT:   %.loc14_36.2: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_36.3: <bound method> = bound_method %.loc14_29, %.loc14_36.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_36.4: <specific function> = specific_function %.loc14_36.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.1: init %i32 = call %.loc14_36.4(%.loc14_29) [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_36.5: init %i32 = converted %.loc14_29, %int.convert_checked.loc14_36.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_36.6: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc14_36.7: init %i32 = initialize_from %.loc14_36.5 to %.loc14_36.6 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_36.8: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_36.9: <bound method> = bound_method %.loc14_32, %.loc14_36.8 [template = constants.%.35]
// CHECK:STDOUT:   %.loc14_36.10: <specific function> = specific_function %.loc14_36.9, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.2: init %i32 = call %.loc14_36.10(%.loc14_32) [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_36.11: init %i32 = converted %.loc14_32, %int.convert_checked.loc14_36.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_36.12: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc14_36.13: init %i32 = initialize_from %.loc14_36.11 to %.loc14_36.12 [template = constants.%.37]
// CHECK:STDOUT:   %.loc14_36.14: %Convert.type.2 = interface_witness_access constants.%.31, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_36.15: <bound method> = bound_method %.loc14_35, %.loc14_36.14 [template = constants.%.38]
// CHECK:STDOUT:   %.loc14_36.16: <specific function> = specific_function %.loc14_36.15, @Convert.2(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.3: init %i32 = call %.loc14_36.16(%.loc14_35) [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_36.17: init %i32 = converted %.loc14_35, %int.convert_checked.loc14_36.3 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_36.18: ref %i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc14_36.19: init %i32 = initialize_from %.loc14_36.17 to %.loc14_36.18 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_36.20: init %tuple.type.3 = tuple_init (%.loc14_36.7, %.loc14_36.13, %.loc14_36.19) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_37: init %tuple.type.3 = converted %.loc14_36.1, %.loc14_36.20 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc14_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
