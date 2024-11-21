// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_todo_assoc_const_default.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_todo_assoc_const_default.carbon

interface I {
  // CHECK:STDERR: fail_todo_assoc_const_default.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default let T:! type = (i32, i32);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let T:! type = (i32, i32);
  // CHECK:STDERR: fail_todo_assoc_const_default.carbon:[[@LINE+3]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default let N:! i32 = 42;
  // CHECK:STDERR:   ^~~~~~~
  default let N:! i32 = 42;
}

// CHECK:STDOUT: --- fail_todo_assoc_const_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 42 [template]
// CHECK:STDOUT:   %.32: type = assoc_entity_type %I.type, %i32 [template]
// CHECK:STDOUT:   %.33: %.32 = assoc_entity element1, @I.%N [template]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %.loc16_27: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc16_27: init type = call constants.%Int(%.loc16_27) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_32: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc16_32: init type = call constants.%Int(%.loc16_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_35: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc16_27, %int.make_type_signed.loc16_32)
// CHECK:STDOUT:   %.loc16_36.1: type = value_of_initializer %int.make_type_signed.loc16_27 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.2: type = converted %int.make_type_signed.loc16_27, %.loc16_36.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.3: type = value_of_initializer %int.make_type_signed.loc16_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.4: type = converted %int.make_type_signed.loc16_32, %.loc16_36.3 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.5: type = converted %.loc16_35, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc16_36.6: %.2 = assoc_entity element0, %T [template = constants.%.3]
// CHECK:STDOUT:   %.loc20_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%.loc20_19.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_19.2: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_19.3: type = converted %int.make_type_signed.loc20, %.loc20_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_25: Core.IntLiteral = int_value 42 [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_27.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc20_27.2: <bound method> = bound_method %.loc20_25, %.loc20_27.1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc20_27.3: <specific function> = specific_function %.loc20_27.2, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc20_27.3(%.loc20_25) [template = constants.%.31]
// CHECK:STDOUT:   %.loc20_27.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.31]
// CHECK:STDOUT:   %.loc20_27.5: %i32 = converted %.loc20_25, %.loc20_27.4 [template = constants.%.31]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %.loc20_27.6: %.32 = assoc_entity element1, %N [template = constants.%.33]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc16_36.6
// CHECK:STDOUT:   .N = %.loc20_27.6
// CHECK:STDOUT:   witness = (%T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
