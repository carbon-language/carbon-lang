// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/basic.carbon

fn A(p: const i32**) -> const i32** {
  return p;
}

fn B(p: const (i32*)) -> const (i32*) {
  return p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const.20a: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.36b: type = ptr_type %const.20a [concrete]
// CHECK:STDOUT:   %ptr.3bc: type = ptr_type %ptr.36b [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %const.12f: type = const_type %ptr.235 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.patt: %ptr.3bc = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.3bc = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.3bc = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.3bc = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %const.loc11_25: type = const_type %i32.loc11_31 [concrete = constants.%const.20a]
// CHECK:STDOUT:     %ptr.loc11_34: type = ptr_type %const.loc11_25 [concrete = constants.%ptr.36b]
// CHECK:STDOUT:     %ptr.loc11_35: type = ptr_type %ptr.loc11_34 [concrete = constants.%ptr.3bc]
// CHECK:STDOUT:     %p.param: %ptr.3bc = value_param call_param0
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr.loc11_19 [concrete = constants.%ptr.3bc] {
// CHECK:STDOUT:       %int_32.loc11_15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc11_15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %const.loc11_9: type = const_type %i32.loc11_15 [concrete = constants.%const.20a]
// CHECK:STDOUT:       %ptr.loc11_18: type = ptr_type %const.loc11_9 [concrete = constants.%ptr.36b]
// CHECK:STDOUT:       %ptr.loc11_19: type = ptr_type %ptr.loc11_18 [concrete = constants.%ptr.3bc]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.3bc = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.3bc = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.3bc = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %p.patt: %const.12f = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %const.12f = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %const.12f = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const.12f = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_36: type = ptr_type %i32.loc15_33 [concrete = constants.%ptr.235]
// CHECK:STDOUT:     %const.loc15_26: type = const_type %ptr.loc15_36 [concrete = constants.%const.12f]
// CHECK:STDOUT:     %p.param: %const.12f = value_param call_param0
// CHECK:STDOUT:     %.loc15: type = splice_block %const.loc15_9 [concrete = constants.%const.12f] {
// CHECK:STDOUT:       %int_32.loc15_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc15_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %ptr.loc15_19: type = ptr_type %i32.loc15_16 [concrete = constants.%ptr.235]
// CHECK:STDOUT:       %const.loc15_9: type = const_type %ptr.loc15_19 [concrete = constants.%const.12f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.12f = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.12f = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.12f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param_patt: %ptr.3bc) -> %ptr.3bc {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.3bc = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param_patt: %const.12f) -> %const.12f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.12f = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
