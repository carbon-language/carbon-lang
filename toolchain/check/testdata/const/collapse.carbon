// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/collapse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/collapse.carbon

// OK, `const (const i32)` is the same type as `const i32`.
// CHECK:STDERR: collapse.carbon:[[@LINE+4]]:25: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn F(p: const i32**) -> const (const i32)** {
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(p: const i32**) -> const (const i32)** {
  return p;
}

// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.36b: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %ptr.3bc: type = ptr_type %ptr.36b [concrete]
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
// CHECK:STDOUT:     %p.patt: %ptr.3bc = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.3bc = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.3bc = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.3bc = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc16_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc16_32: type = const_type %i32.loc16_38 [concrete = constants.%const]
// CHECK:STDOUT:     %const.loc16_25: type = const_type %const.loc16_32 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr.loc16_42: type = ptr_type %const.loc16_25 [concrete = constants.%ptr.36b]
// CHECK:STDOUT:     %ptr.loc16_43: type = ptr_type %ptr.loc16_42 [concrete = constants.%ptr.3bc]
// CHECK:STDOUT:     %p.param: %ptr.3bc = value_param call_param0
// CHECK:STDOUT:     %.loc16: type = splice_block %ptr.loc16_19 [concrete = constants.%ptr.3bc] {
// CHECK:STDOUT:       %int_32.loc16_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc16_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc16_9: type = const_type %i32.loc16_15 [concrete = constants.%const]
// CHECK:STDOUT:       %ptr.loc16_18: type = ptr_type %const.loc16_9 [concrete = constants.%ptr.36b]
// CHECK:STDOUT:       %ptr.loc16_19: type = ptr_type %ptr.loc16_18 [concrete = constants.%ptr.3bc]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.3bc = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.3bc = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.3bc = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param_patt: %ptr.3bc) -> %ptr.3bc {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.3bc = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
