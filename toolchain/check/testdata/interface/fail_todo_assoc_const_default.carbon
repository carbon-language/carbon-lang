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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %assoc_type.1: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type.1 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %int_42.1: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_42.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_42.2: %i32 = int_value 42 [template]
// CHECK:STDOUT:   %assoc_type.2: type = assoc_entity_type %I.type, %i32 [template]
// CHECK:STDOUT:   %assoc1: %assoc_type.2 = assoc_entity element1, @I.%N [template]
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
// CHECK:STDOUT:   %int_32.loc16_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16_27: init type = call constants.%Int(%int_32.loc16_27) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc16_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16_32: init type = call constants.%Int(%int_32.loc16_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_35: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc16_27, %int.make_type_signed.loc16_32)
// CHECK:STDOUT:   %.loc16_36.1: type = value_of_initializer %int.make_type_signed.loc16_27 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.2: type = converted %int.make_type_signed.loc16_27, %.loc16_36.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.3: type = value_of_initializer %int.make_type_signed.loc16_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.4: type = converted %int.make_type_signed.loc16_32, %.loc16_36.3 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_36.5: type = converted %.loc16_35, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.1 = assoc_entity element0, %T [template = constants.%assoc0.1]
// CHECK:STDOUT:   %int_32.loc20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%int_32.loc20) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_19.1: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_19.2: type = converted %int.make_type_signed.loc20, %.loc20_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [template = constants.%int_42.1]
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_42, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_42) [template = constants.%int_42.2]
// CHECK:STDOUT:   %.loc20_27.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_42.2]
// CHECK:STDOUT:   %.loc20_27.2: %i32 = converted %int_42, %.loc20_27.1 [template = constants.%int_42.2]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %assoc1: %assoc_type.2 = assoc_entity element1, %N [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %assoc0
// CHECK:STDOUT:   .N = %assoc1
// CHECK:STDOUT:   witness = (%T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
