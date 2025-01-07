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
// CHECK:STDOUT:   %Print.type.980d: type = fn_type @Print.1 [template]
// CHECK:STDOUT:   %Print.b7c5: %Print.type.980d = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd1: <bound method> = bound_method %int_1.5b81, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b730: <specific function> = specific_function %Convert.bound.afd1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c605: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Print.type.6ed1: type = fn_type @Print.2 [template]
// CHECK:STDOUT:   %Print.7238: %Print.type.6ed1 = struct_value () [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f0b0: <bound method> = bound_method %int_2.ecc8, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e1: <specific function> = specific_function %Convert.bound.f0b0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.1667: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     .Print = %import_ref.ace3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//io
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.ace3: %Print.type.6ed1 = import_ref Core//io, Print, loaded [template = constants.%Print.7238]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Print = %Print.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Print.decl: %Print.type.980d = fn_decl @Print.1 [template = constants.%Print.b7c5] {
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
// CHECK:STDOUT:   %Print.ref.loc16: %Print.type.980d = name_ref Print, file.%Print.decl [template = constants.%Print.b7c5]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %impl.elem0.loc16: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc16: <bound method> = bound_method %int_1, %impl.elem0.loc16 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:   %Convert.specific_fn.loc16: <specific function> = specific_function %Convert.bound.loc16, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %Convert.specific_fn.loc16(%int_1) [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc16_9.1: %i32 = value_of_initializer %int.convert_checked.loc16 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc16_9.2: %i32 = converted %int_1, %.loc16_9.1 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %print.int.loc16: init %empty_tuple.type = call %Print.ref.loc16(%.loc16_9.2)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %Print.ref.loc18: %Print.type.6ed1 = name_ref Print, imports.%import_ref.ace3 [template = constants.%Print.7238]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %impl.elem0.loc18: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc18: <bound method> = bound_method %int_2, %impl.elem0.loc18 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:   %Convert.specific_fn.loc18: <specific function> = specific_function %Convert.bound.loc18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %Convert.specific_fn.loc18(%int_2) [template = constants.%int_2.1667]
// CHECK:STDOUT:   %.loc18_14.1: %i32 = value_of_initializer %int.convert_checked.loc18 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %.loc18_14.2: %i32 = converted %int_2, %.loc18_14.1 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %print.int.loc18: init %empty_tuple.type = call %Print.ref.loc18(%.loc18_14.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
