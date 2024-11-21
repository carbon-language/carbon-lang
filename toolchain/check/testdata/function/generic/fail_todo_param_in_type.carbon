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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.13: type = fn_type @Convert.4, @impl.3(%.1) [template]
// CHECK:STDOUT:   %Convert.13: %Convert.type.13 = struct_value () [template]
// CHECK:STDOUT:   %.23: <witness> = interface_witness (%Convert.13) [template]
// CHECK:STDOUT:   %.24: <bound method> = bound_method %N.1, %Convert.13 [symbolic]
// CHECK:STDOUT:   %.25: <specific function> = specific_function %.24, @Convert.4(%.1) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.25(%N.1) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc14_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc14_6.1, runtime_param<invalid> [symbolic = %N.patt.loc14_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc14_10: init type = call constants.%Int(%.loc14_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_10.2: type = value_of_initializer %int.make_type_signed.loc14_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_10.3: type = converted %int.make_type_signed.loc14_10, %.loc14_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc14_19: init type = call constants.%Int(%.loc14_19.1) [template = constants.%i32]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %N.loc14_6.1 [symbolic = %N.loc14_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %.loc14_19.2: type = value_of_initializer %int.make_type_signed.loc14_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_19.3: type = converted %int.make_type_signed.loc14_19, %.loc14_19.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_24.1: %Convert.type.2 = interface_witness_access constants.%.23, element0 [template = constants.%Convert.13]
// CHECK:STDOUT:     %.loc14_24.2: <bound method> = bound_method %N.ref, %.loc14_24.1 [symbolic = %.loc14_24.6 (constants.%.24)]
// CHECK:STDOUT:     %.loc14_24.3: <specific function> = specific_function %.loc14_24.2, @Convert.4(constants.%.1) [symbolic = %.loc14_24.7 (constants.%.25)]
// CHECK:STDOUT:     %int.convert_checked.loc14_24.1: init Core.IntLiteral = call %.loc14_24.3(%N.ref) [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc14_24.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_24.1 [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc14_24.5: Core.IntLiteral = converted %N.ref, %.loc14_24.4 [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc14_25: type = array_type %.loc14_24.5, %i32 [template = <error>]
// CHECK:STDOUT:     %.loc14_26: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc14_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc14_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc14_6.1: %i32) {
// CHECK:STDOUT:   %N.loc14_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc14_6.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc14_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc14_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   %.loc14_24.6: <bound method> = bound_method %N.loc14_6.2, constants.%Convert.13 [symbolic = %.loc14_24.6 (constants.%.24)]
// CHECK:STDOUT:   %.loc14_24.7: <specific function> = specific_function %.loc14_24.6, @Convert.4(constants.%.1) [symbolic = %.loc14_24.7 (constants.%.25)]
// CHECK:STDOUT:   %int.convert_checked.loc14_24.2: init Core.IntLiteral = call %.loc14_24.7(%N.loc14_6.2) [symbolic = %int.convert_checked.loc14_24.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32, %a.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.1) {
// CHECK:STDOUT:   %N.loc14_6.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc14_6.2 => constants.%N.1
// CHECK:STDOUT:   %.loc14_24.6 => constants.%.24
// CHECK:STDOUT:   %.loc14_24.7 => constants.%.25
// CHECK:STDOUT:   %int.convert_checked.loc14_24.2 => constants.%int.convert_checked
// CHECK:STDOUT: }
// CHECK:STDOUT:
