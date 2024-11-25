// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_param_in_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_param_in_type.carbon

// CHECK:STDERR: fail_param_in_type.carbon:[[@LINE+3]]:23: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: fn F(n: i32, a: [i32; n]*);
// CHECK:STDERR:                       ^
fn F(n: i32, a: [i32; n]*);

// CHECK:STDOUT: --- fail_param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
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
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: %i32 = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: <error> = value_param_pattern %a.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc14_9: init type = call constants.%Int(%.loc14_9.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_9.2: type = value_of_initializer %int.make_type_signed.loc14_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_9.3: type = converted %int.make_type_signed.loc14_9, %.loc14_9.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc14_18: init type = call constants.%Int(%.loc14_18.1) [template = constants.%i32]
// CHECK:STDOUT:     %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:     %.loc14_18.2: type = value_of_initializer %int.make_type_signed.loc14_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_18.3: type = converted %int.make_type_signed.loc14_18, %.loc14_18.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_25: type = ptr_type <error> [template = <error>]
// CHECK:STDOUT:     %n.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n: %i32 = bind_name n, %n.param
// CHECK:STDOUT:     %a.param: <error> = value_param runtime_param1
// CHECK:STDOUT:     %a: <error> = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt: %i32, %a.param_patt: <error>);
// CHECK:STDOUT:
