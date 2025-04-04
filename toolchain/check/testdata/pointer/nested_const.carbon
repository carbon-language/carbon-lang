// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/nested_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/nested_const.carbon

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
  return **p;
}

// CHECK:STDOUT: --- nested_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const.20a: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.36b: type = ptr_type %const.20a [concrete]
// CHECK:STDOUT:   %const.58f: type = const_type %ptr.36b [concrete]
// CHECK:STDOUT:   %ptr.6e8: type = ptr_type %const.58f [concrete]
// CHECK:STDOUT:   %const.fa2: type = const_type %ptr.6e8 [concrete]
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
// CHECK:STDOUT:     %p.patt: %const.fa2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %const.fa2 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %const.20a = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const.20a = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_47: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_47: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc12_41: type = const_type %i32.loc12_47 [concrete = constants.%const.20a]
// CHECK:STDOUT:     %p.param: %const.fa2 = value_param call_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %const.loc12_9 [concrete = constants.%const.fa2] {
// CHECK:STDOUT:       %int_32.loc12_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc12_29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc12_23: type = const_type %i32.loc12_29 [concrete = constants.%const.20a]
// CHECK:STDOUT:       %ptr.loc12_32: type = ptr_type %const.loc12_23 [concrete = constants.%ptr.36b]
// CHECK:STDOUT:       %const.loc12_16: type = const_type %ptr.loc12_32 [concrete = constants.%const.58f]
// CHECK:STDOUT:       %ptr.loc12_34: type = ptr_type %const.loc12_16 [concrete = constants.%ptr.6e8]
// CHECK:STDOUT:       %const.loc12_9: type = const_type %ptr.loc12_34 [concrete = constants.%const.fa2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.fa2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.20a = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.20a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param_patt: %const.fa2) -> %const.20a {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.fa2 = name_ref p, %p
// CHECK:STDOUT:   %.loc13_11.1: ref %const.58f = deref %p.ref
// CHECK:STDOUT:   %.loc13_11.2: %const.58f = bind_value %.loc13_11.1
// CHECK:STDOUT:   %.loc13_10.1: ref %const.20a = deref %.loc13_11.2
// CHECK:STDOUT:   %.loc13_10.2: %const.20a = bind_value %.loc13_10.1
// CHECK:STDOUT:   return %.loc13_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
