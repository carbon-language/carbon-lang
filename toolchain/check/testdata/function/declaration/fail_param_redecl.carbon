// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_param_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_param_redecl.carbon

// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+7]]:14: error: duplicate name `n` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+4]]:6: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(n: i32, n: i32);

// CHECK:STDOUT: --- fail_param_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %n.patt.loc18_6: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt.loc18_7: %i32 = value_param_pattern %n.patt.loc18_6, call_param0
// CHECK:STDOUT:     %n.patt.loc18_14: %i32 = binding_pattern n
// CHECK:STDOUT:     %n.param_patt.loc18_15: %i32 = value_param_pattern %n.patt.loc18_14, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param.loc18_7: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc18_9: type = splice_block %i32.loc18_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.loc18_6: %i32 = bind_name n, %n.param.loc18_7
// CHECK:STDOUT:     %n.param.loc18_15: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc18_17: type = splice_block %i32.loc18_17 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.loc18_14: %i32 = bind_name n, %n.param.loc18_15
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.param_patt.loc18_7: %i32, %n.param_patt.loc18_15: %i32);
// CHECK:STDOUT:
