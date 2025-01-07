// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/print/char.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/print/char.carbon

import Core library "io";

fn PrintChar(a: i32) -> i32 = "print.char";

fn Main() {
  PrintChar(1);
  Core.PrintChar(2);
}

// CHECK:STDOUT: --- char.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %PrintChar.type.c: type = fn_type @PrintChar.1 [template]
// CHECK:STDOUT:   %PrintChar.8: %PrintChar.type.c = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.cd: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.46: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.46 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.a: <bound method> = bound_method %int_1.5, %Convert.9 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b: <specific function> = specific_function %Convert.bound.a, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %PrintChar.type.0: type = fn_type @PrintChar.2 [template]
// CHECK:STDOUT:   %PrintChar.d: %PrintChar.type.0 = struct_value () [template]
// CHECK:STDOUT:   %int_2.e: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.f: <bound method> = bound_method %int_2.e, %Convert.9 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.f, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.1: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a6
// CHECK:STDOUT:     .PrintChar = %import_ref.d1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//io
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.d1: %PrintChar.type.0 = import_ref Core//io, PrintChar, loaded [template = constants.%PrintChar.d]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PrintChar = %PrintChar.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PrintChar.decl: %PrintChar.type.c = fn_decl @PrintChar.1 [template = constants.%PrintChar.8] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %i32.loc13_17 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PrintChar.1(%a.param_patt: %i32) -> %i32 = "print.char";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %PrintChar.ref.loc16: %PrintChar.type.c = name_ref PrintChar, file.%PrintChar.decl [template = constants.%PrintChar.8]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5]
// CHECK:STDOUT:   %impl.elem0.loc16: %Convert.type.cd = impl_witness_access constants.%impl_witness.5, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %Convert.bound.loc16: <bound method> = bound_method %int_1, %impl.elem0.loc16 [template = constants.%Convert.bound.a]
// CHECK:STDOUT:   %Convert.specific_fn.loc16: <specific function> = specific_function %Convert.bound.loc16, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %Convert.specific_fn.loc16(%int_1) [template = constants.%int_1.c]
// CHECK:STDOUT:   %.loc16_13.1: %i32 = value_of_initializer %int.convert_checked.loc16 [template = constants.%int_1.c]
// CHECK:STDOUT:   %.loc16_13.2: %i32 = converted %int_1, %.loc16_13.1 [template = constants.%int_1.c]
// CHECK:STDOUT:   %print.char.loc16: init %i32 = call %PrintChar.ref.loc16(%.loc16_13.2)
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:   %PrintChar.ref.loc17: %PrintChar.type.0 = name_ref PrintChar, imports.%import_ref.d1 [template = constants.%PrintChar.d]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.e]
// CHECK:STDOUT:   %impl.elem0.loc17: %Convert.type.cd = impl_witness_access constants.%impl_witness.5, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:   %Convert.bound.loc17: <bound method> = bound_method %int_2, %impl.elem0.loc17 [template = constants.%Convert.bound.f]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %Convert.bound.loc17, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %Convert.specific_fn.loc17(%int_2) [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc17_18.1: %i32 = value_of_initializer %int.convert_checked.loc17 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc17_18.2: %i32 = converted %int_2, %.loc17_18.1 [template = constants.%int_2.1]
// CHECK:STDOUT:   %print.char.loc17: init %i32 = call %PrintChar.ref.loc17(%.loc17_18.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
