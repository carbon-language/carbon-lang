// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/print/int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/print/int.carbon

import Core library "io";

fn Print(a: i32) = "print.int";

fn Main() {
  Print(1);

  Core.Print(2);
}

// CHECK:STDOUT: --- int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Print.type.980: type = fn_type @Print.1 [template]
// CHECK:STDOUT:   %Print.b7c: %Print.type.980 = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [template]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [template]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn.70c: <specific function> = specific_function %Convert.bound.ab5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Print.type.6ed: type = fn_type @Print.2 [template]
// CHECK:STDOUT:   %Print.723: %Print.type.6ed = struct_value () [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn.787: <specific function> = specific_function %Convert.bound.ef9, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Print = %Core.Print
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//io
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Print: %Print.type.6ed = import_ref Core//io, Print, loaded [template = constants.%Print.723]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Print = %Print.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Print.decl: %Print.type.980 = fn_decl @Print.1 [template = constants.%Print.b7c] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Print.1(%a.param_patt: %i32) = "print.int";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Print.ref.loc16: %Print.type.980 = name_ref Print, file.%Print.decl [template = constants.%Print.b7c]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %int_1, %impl.elem0.loc16 [template = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %bound_method.loc16, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.70c]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %specific_fn.loc16(%int_1) [template = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_9.1: %i32 = value_of_initializer %int.convert_checked.loc16 [template = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_9.2: %i32 = converted %int_1, %.loc16_9.1 [template = constants.%int_1.5d2]
// CHECK:STDOUT:   %print.int.loc16: init %empty_tuple.type = call %Print.ref.loc16(%.loc16_9.2)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %Print.ref.loc18: %Print.type.6ed = name_ref Print, imports.%Core.Print [template = constants.%Print.723]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc18: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18: <bound method> = bound_method %int_2, %impl.elem0.loc18 [template = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %bound_method.loc18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.787]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %specific_fn.loc18(%int_2) [template = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_14.1: %i32 = value_of_initializer %int.convert_checked.loc18 [template = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_14.2: %i32 = converted %int_2, %.loc18_14.1 [template = constants.%int_2.ef8]
// CHECK:STDOUT:   %print.int.loc18: init %empty_tuple.type = call %Print.ref.loc18(%.loc18_14.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
