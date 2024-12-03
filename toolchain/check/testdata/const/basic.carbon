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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %const.1: type = const_type %i32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %const.1 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %ptr.1 [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %const.2: type = const_type %ptr.3 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_15: init type = call constants.%Int(%int_32.loc11_15) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11_15 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_9.2: type = converted %int.make_type_signed.loc11_15, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %const.loc11_9: type = const_type %i32 [template = constants.%const.1]
// CHECK:STDOUT:     %ptr.loc11_18: type = ptr_type %const.1 [template = constants.%ptr.1]
// CHECK:STDOUT:     %ptr.loc11_19: type = ptr_type %ptr.1 [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32.loc11_31: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_31: init type = call constants.%Int(%int_32.loc11_31) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: type = value_of_initializer %int.make_type_signed.loc11_31 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.2: type = converted %int.make_type_signed.loc11_31, %.loc11_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %const.loc11_25: type = const_type %i32 [template = constants.%const.1]
// CHECK:STDOUT:     %ptr.loc11_34: type = ptr_type %const.1 [template = constants.%ptr.1]
// CHECK:STDOUT:     %ptr.loc11_35: type = ptr_type %ptr.1 [template = constants.%ptr.2]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %p.patt: %const.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %const.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %const.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %const.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_16: init type = call constants.%Int(%int_32.loc15_16) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_19.1: type = value_of_initializer %int.make_type_signed.loc15_16 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_19.2: type = converted %int.make_type_signed.loc15_16, %.loc15_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_19: type = ptr_type %i32 [template = constants.%ptr.3]
// CHECK:STDOUT:     %const.loc15_9: type = const_type %ptr.3 [template = constants.%const.2]
// CHECK:STDOUT:     %int_32.loc15_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_33: init type = call constants.%Int(%int_32.loc15_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_signed.loc15_33 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_signed.loc15_33, %.loc15_36.1 [template = constants.%i32]
// CHECK:STDOUT:     %ptr.loc15_36: type = ptr_type %i32 [template = constants.%ptr.3]
// CHECK:STDOUT:     %const.loc15_26: type = const_type %ptr.3 [template = constants.%const.2]
// CHECK:STDOUT:     %p.param: %const.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %const.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %const.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param_patt: %ptr.2) -> %ptr.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param_patt: %const.2) -> %const.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.2 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
