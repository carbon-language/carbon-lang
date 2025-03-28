// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/fail_collapse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/fail_collapse.carbon

// CHECK:STDERR: fail_collapse.carbon:[[@LINE+4]]:9: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn G(p: const (const i32)**) -> i32** {
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: const (const i32)**) -> i32** {
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+7]]:3: error: cannot implicitly convert value of type `const i32**` to `i32**` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+4]]:3: note: type `const i32**` does not implement interface `Core.ImplicitAs(i32**)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// CHECK:STDOUT: --- fail_collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.36b: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %ptr.3bc: type = ptr_type %ptr.36b [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.5d5: type = ptr_type %ptr.235 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %ptr.3bc = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.3bc = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.5d5 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.5d5 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_36: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %ptr.loc15_37: type = ptr_type %ptr.235 [concrete = constants.%ptr.5d5]
// CHECK:STDOUT:     %p.param: %ptr.3bc = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %ptr.loc15_27 [concrete = constants.%ptr.3bc] {
// CHECK:STDOUT:       %int_32.loc15_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc15_16: type = const_type %i32 [concrete = constants.%const]
// CHECK:STDOUT:       %const.loc15_9: type = const_type %const [concrete = constants.%const]
// CHECK:STDOUT:       %ptr.loc15_26: type = ptr_type %const [concrete = constants.%ptr.36b]
// CHECK:STDOUT:       %ptr.loc15_27: type = ptr_type %ptr.36b [concrete = constants.%ptr.3bc]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.3bc = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.5d5 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.5d5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %ptr.3bc) -> %ptr.5d5 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.3bc = name_ref p, %p
// CHECK:STDOUT:   %.loc23: %ptr.5d5 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
