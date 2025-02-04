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
  // CHECK:STDERR: fail_todo_assoc_const_default.carbon:[[@LINE+4]]:3: error: semantics TODO: `interface modifier` [SemanticsTodo]
  // CHECK:STDERR:   default let N:! i32 = 42;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let N:! i32 = 42;
}

// CHECK:STDOUT: --- fail_todo_assoc_const_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [template]
// CHECK:STDOUT:   %assoc0.c7e: %I.assoc_type = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%N [template]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [template]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_42.20e, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [template] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [template = constants.%assoc0.c7e]
// CHECK:STDOUT:     %int_32.loc16_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc16_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc16_35: %tuple.type.24b = tuple_literal (%i32.loc16_27, %i32.loc16_32)
// CHECK:STDOUT:     %.loc16_36: type = converted %.loc16_35, constants.%tuple.type.d07 [template = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [template] {
// CHECK:STDOUT:     %assoc1: %I.assoc_type = assoc_entity element1, @I.%N [template = constants.%assoc1]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [template = constants.%int_42.20e]
// CHECK:STDOUT:     %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_42, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %bound_method, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %specific_fn(%int_42) [template = constants.%int_42.c68]
// CHECK:STDOUT:     %.loc21_27.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_42.c68]
// CHECK:STDOUT:     %.loc21_27.2: %i32 = converted %int_42, %.loc21_27.1 [template = constants.%int_42.c68]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .N = @N.%assoc1
// CHECK:STDOUT:   witness = (%T, %N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const T:! type = %.loc16_36;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @N(@I.%Self: %I.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   assoc_const N:! %i32 = %.loc21_27.2;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Self.826) {}
// CHECK:STDOUT:
