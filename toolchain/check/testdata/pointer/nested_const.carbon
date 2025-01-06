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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %const.f: type = const_type %i32 [template]
// CHECK:STDOUT:   %ptr.8: type = ptr_type %const.f [template]
// CHECK:STDOUT:   %const.1e: type = const_type %ptr.8 [template]
// CHECK:STDOUT:   %ptr.f: type = ptr_type %const.1e [template]
// CHECK:STDOUT:   %const.15: type = const_type %ptr.f [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.3
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
// CHECK:STDOUT:     %p.patt: %const.15 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %const.15 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %const.f = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const.f = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12_47: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_47: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %const.loc12_41: type = const_type %i32 [template = constants.%const.f]
// CHECK:STDOUT:     %p.param: %const.15 = value_param runtime_param0
// CHECK:STDOUT:     %.loc12: type = splice_block %const.loc12_9 [template = constants.%const.15] {
// CHECK:STDOUT:       %int_32.loc12_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc12_29: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %const.loc12_23: type = const_type %i32 [template = constants.%const.f]
// CHECK:STDOUT:       %ptr.loc12_32: type = ptr_type %const.f [template = constants.%ptr.8]
// CHECK:STDOUT:       %const.loc12_16: type = const_type %ptr.8 [template = constants.%const.1e]
// CHECK:STDOUT:       %ptr.loc12_34: type = ptr_type %const.1e [template = constants.%ptr.f]
// CHECK:STDOUT:       %const.loc12_9: type = const_type %ptr.f [template = constants.%const.15]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.15 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.f = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %const.f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param_patt: %const.15) -> %const.f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.15 = name_ref p, %p
// CHECK:STDOUT:   %.loc13_11.1: ref %const.1e = deref %p.ref
// CHECK:STDOUT:   %.loc13_11.2: %const.1e = bind_value %.loc13_11.1
// CHECK:STDOUT:   %.loc13_10.1: ref %const.f = deref %.loc13_11.2
// CHECK:STDOUT:   %.loc13_10.2: %const.f = bind_value %.loc13_10.1
// CHECK:STDOUT:   return %.loc13_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
