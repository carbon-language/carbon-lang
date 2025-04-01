// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_generic.carbon

// TODO: Should this be valid?
fn F(a: i32) -> i32 {
  let T:! type = i32;
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+7]]:14: error: cannot implicitly convert expression of type `Core.IntLiteral` to `T` [ConversionFailure]
  // CHECK:STDERR:   let x: T = 5;
  // CHECK:STDERR:              ^
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+4]]:14: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(T)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let x: T = 5;
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  let x: T = 5;
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `T` to `i32` [ConversionFailure]
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+4]]:3: note: type `T` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return x;
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %i32.loc12_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc12_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc12_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = constants.%T.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, %i32.loc13 [symbolic = constants.%T]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %T = binding_pattern x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc21: %T = converted %int_5, <error> [concrete = <error>]
// CHECK:STDOUT:   %x: %T = bind_name x, <error>
// CHECK:STDOUT:   %x.ref: %T = name_ref x, %x
// CHECK:STDOUT:   %.loc29: %i32 = converted %x.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
