// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_param_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_param_redecl.carbon

// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+6]]:14: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+3]]:6: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:      ^
fn F(n: i32, n: i32);

// CHECK:STDOUT: --- fail_param_redecl.carbon
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
// CHECK:STDOUT:     %n.patt.loc17_6: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt.loc17_7: %i32 = value_param_pattern %n.patt.loc17_6, runtime_param0
// CHECK:STDOUT:     %n.patt.loc17_14: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt.loc17_15: %i32 = value_param_pattern %n.patt.loc17_14, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc17_9: init type = call constants.%Int(%.loc17_9.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_9.2: type = value_of_initializer %int.make_type_signed.loc17_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_9.3: type = converted %int.make_type_signed.loc17_9, %.loc17_9.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_17.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc17_17: init type = call constants.%Int(%.loc17_17.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_17.2: type = value_of_initializer %int.make_type_signed.loc17_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc17_17.3: type = converted %int.make_type_signed.loc17_17, %.loc17_17.2 [template = constants.%i32]
// CHECK:STDOUT:     %n.param.loc17_7: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %n.loc17_6: %i32 = bind_name n, %n.param.loc17_7
// CHECK:STDOUT:     %n.param.loc17_15: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %n.loc17_14: %i32 = bind_name n, %n.param.loc17_15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt.loc17_7: %i32, %n.param_patt.loc17_15: %i32);
// CHECK:STDOUT:
