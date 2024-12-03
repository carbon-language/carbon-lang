// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/types.carbon

fn Ptr(p: i32*) -> i32* {
  return p;
}

fn ConstPtr(p: const i32*) -> (const i32)* {
  return p;
}

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %Ptr.type: type = fn_type @Ptr [template]
// CHECK:STDOUT:   %Ptr: %Ptr.type = struct_value () [template]
// CHECK:STDOUT:   %const: type = const_type %i32 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %const [template]
// CHECK:STDOUT:   %ConstPtr.type: type = fn_type @ConstPtr [template]
// CHECK:STDOUT:   %ConstPtr: %ConstPtr.type = struct_value () [template]
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
// CHECK:STDOUT:     .Ptr = %Ptr.decl
// CHECK:STDOUT:     .ConstPtr = %ConstPtr.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Ptr.decl: %Ptr.type = fn_decl @Ptr [template = constants.%Ptr] {
// CHECK:STDOUT:     %p.patt: %ptr.1 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.1 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.1 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_11: init type = call constants.%Int(%int_32.loc11_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.1: type = value_of_initializer %int.make_type_signed.loc11_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_14.2: type = converted %int.make_type_signed.loc11_11, %.loc11_14.1 [template = constants.%i32]
// CHECK:STDOUT:     %ptr.loc11_14: type = ptr_type %i32 [template = constants.%ptr.1]
// CHECK:STDOUT:     %int_32.loc11_20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_20: init type = call constants.%Int(%int_32.loc11_20) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_23.1: type = value_of_initializer %int.make_type_signed.loc11_20 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_23.2: type = converted %int.make_type_signed.loc11_20, %.loc11_23.1 [template = constants.%i32]
// CHECK:STDOUT:     %ptr.loc11_23: type = ptr_type %i32 [template = constants.%ptr.1]
// CHECK:STDOUT:     %p.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.1 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.1 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr.decl: %ConstPtr.type = fn_decl @ConstPtr [template = constants.%ConstPtr] {
// CHECK:STDOUT:     %p.patt: %ptr.2 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.2 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %ptr.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.2 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_22: init type = call constants.%Int(%int_32.loc15_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_16.1: type = value_of_initializer %int.make_type_signed.loc15_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_16.2: type = converted %int.make_type_signed.loc15_22, %.loc15_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %const.loc15_16: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:     %ptr.loc15_25: type = ptr_type %const [template = constants.%ptr.2]
// CHECK:STDOUT:     %int_32.loc15_38: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc15_38: init type = call constants.%Int(%int_32.loc15_38) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %int.make_type_signed.loc15_38 [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %int.make_type_signed.loc15_38, %.loc15_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %const.loc15_32: type = const_type %i32 [template = constants.%const]
// CHECK:STDOUT:     %ptr.loc15_42: type = ptr_type %const [template = constants.%ptr.2]
// CHECK:STDOUT:     %p.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %p: %ptr.2 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.2 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %ptr.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p.param_patt: %ptr.1) -> %ptr.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.1 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p.param_patt: %ptr.2) -> %ptr.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.2 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
