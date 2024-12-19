// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/fail_todo_param_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/fail_todo_param_in_type.carbon

// CHECK:STDERR: fail_todo_param_in_type.carbon:[[@LINE+3]]:24: error: semantics TODO: `symbolic array bound` [SemanticsTodo]
// CHECK:STDERR: fn F(N:! i32, a: [i32; N]*);
// CHECK:STDERR:                        ^
fn F(N:! i32, a: [i32; N]*);

// CHECK:STDOUT: --- fail_todo_param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %N.2: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.9: %Convert.type.9 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.9) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.2, %Convert.9 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.2) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc14_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc14_6.1, runtime_param<invalid> [symbolic = %N.patt.loc14_6.2 (constants.%N.patt.2)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc14_10: type = splice_block %i32.loc14_10 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc14_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_6.2 (constants.%N.2)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %.loc14_26: type = splice_block %ptr [template = <error>] {
// CHECK:STDOUT:       %int_32.loc14_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc14_6.1 [symbolic = %N.loc14_6.2 (constants.%N.2)]
// CHECK:STDOUT:       %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.9]
// CHECK:STDOUT:       %Convert.bound.loc14_24.1: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound.loc14_24.2 (constants.%Convert.bound)]
// CHECK:STDOUT:       %Convert.specific_fn.loc14_24.1: <specific function> = specific_function %Convert.bound.loc14_24.1, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn.loc14_24.2 (constants.%Convert.specific_fn)]
// CHECK:STDOUT:       %int.convert_checked.loc14_24.1: init Core.IntLiteral = call %Convert.specific_fn.loc14_24.1(%N.ref) [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc14_24.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_24.1 [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc14_24.2: Core.IntLiteral = converted %N.ref, %.loc14_24.1 [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type: type = array_type %.loc14_24.2, %i32 [template = <error>]
// CHECK:STDOUT:       %ptr: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc14_6.1: %i32) {
// CHECK:STDOUT:   %N.loc14_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc14_6.2 (constants.%N.2)]
// CHECK:STDOUT:   %N.patt.loc14_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt.2)]
// CHECK:STDOUT:   %Convert.bound.loc14_24.2: <bound method> = bound_method %N.loc14_6.2, constants.%Convert.9 [symbolic = %Convert.bound.loc14_24.2 (constants.%Convert.bound)]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_24.2: <specific function> = specific_function %Convert.bound.loc14_24.2, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn.loc14_24.2 (constants.%Convert.specific_fn)]
// CHECK:STDOUT:   %int.convert_checked.loc14_24.2: init Core.IntLiteral = call %Convert.specific_fn.loc14_24.2(%N.loc14_6.2) [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32, %a.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.2) {
// CHECK:STDOUT:   %N.loc14_6.2 => constants.%N.2
// CHECK:STDOUT:   %N.patt.loc14_6.2 => constants.%N.2
// CHECK:STDOUT:   %Convert.bound.loc14_24.2 => constants.%Convert.bound
// CHECK:STDOUT:   %Convert.specific_fn.loc14_24.2 => constants.%Convert.specific_fn
// CHECK:STDOUT:   %int.convert_checked.loc14_24.2 => constants.%int.convert_checked
// CHECK:STDOUT: }
// CHECK:STDOUT:
