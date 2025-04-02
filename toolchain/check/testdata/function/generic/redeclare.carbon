// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/redeclare.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/redeclare.carbon

// --- redeclare.carbon

library "[[@TEST_NAME]]";

fn F(T:! type) -> T*;

fn F(T:! type) -> T* {
  return F(T);
}

// --- fail_different_return_type.carbon

library "[[@TEST_NAME]]";

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_different_return_type.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `U*` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn F(T:! type, U:! type) -> U* {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_different_return_type.carbon:[[@LINE-5]]:1: note: previously declared with return type `T*` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! type, U:! type) -> U* {
  // CHECK:STDERR: fail_different_return_type.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_different_return_type.carbon:[[@LINE-13]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(T);
}

// --- fail_reorder.carbon

library "[[@TEST_NAME]]";

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_reorder.carbon:[[@LINE+7]]:6: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fn F(U:! type, T:! type) -> T* {
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_reorder.carbon:[[@LINE-5]]:6: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(U:! type, T:! type) -> T* {
  // CHECK:STDERR: fail_reorder.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_reorder.carbon:[[@LINE-13]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(T);
}

// --- fail_rename.carbon

library "[[@TEST_NAME]]";

fn F(T:! type, U:! type) -> T*;

// CHECK:STDERR: fail_rename.carbon:[[@LINE+7]]:6: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fn F(U:! type, T:! type) -> U* {
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_rename.carbon:[[@LINE-5]]:6: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(U:! type, T:! type) -> U* {
  // CHECK:STDERR: fail_rename.carbon:[[@LINE+7]]:10: error: 1 argument passed to function expecting 2 arguments [CallArgCountMismatch]
  // CHECK:STDERR:   return F(T);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_rename.carbon:[[@LINE-13]]:1: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(T);
}

// CHECK:STDOUT: --- redeclare.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc4: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @F.%ptr.loc4_20.2 (%ptr) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%ptr.loc4_20.2 (%ptr) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc4_20.1: type = ptr_type %T [symbolic = %ptr.loc4_20.2 (constants.%ptr)]
// CHECK:STDOUT:     %T.loc4_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %return.param.loc4: ref @F.%ptr.loc4_20.2 (%ptr) = out_param call_param0
// CHECK:STDOUT:     %return.loc4: ref @F.%ptr.loc4_20.2 (%ptr) = return_slot %return.param.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc6: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:     %return.patt: @F.%ptr.loc4_20.2 (%ptr) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%ptr.loc4_20.2 (%ptr) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6: type = name_ref T, %T.loc6 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc6: type = ptr_type %T [symbolic = %ptr.loc4_20.2 (constants.%ptr)]
// CHECK:STDOUT:     %T.loc6: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %return.param.loc6: ref @F.%ptr.loc4_20.2 (%ptr) = out_param call_param0
// CHECK:STDOUT:     %return.loc6: ref @F.%ptr.loc4_20.2 (%ptr) = return_slot %return.param.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.1: type) {
// CHECK:STDOUT:   %T.loc4_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc4_20.2: type = ptr_type @F.%T.loc4_6.2 (%T) [symbolic = %ptr.loc4_20.2 (constants.%ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%ptr.loc4_20.2 (%ptr) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_10.2: <specific function> = specific_function constants.%F, @F(%T.loc4_6.2) [symbolic = %F.specific_fn.loc7_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc6: type) -> @F.%ptr.loc4_20.2 (%ptr) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl.loc4 [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc7: type = name_ref T, %T.loc6 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %F.specific_fn.loc7_10.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init @F.%ptr.loc4_20.2 (%ptr) = call %F.specific_fn.loc7_10.1()
// CHECK:STDOUT:     %.loc7_14.1: @F.%ptr.loc4_20.2 (%ptr) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc7_14.2: @F.%ptr.loc4_20.2 (%ptr) = converted %F.call, %.loc7_14.1
// CHECK:STDOUT:     return %.loc7_14.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc4_20.2 => constants.%ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_10.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc4_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_different_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %F.type.b25846.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41931.1: %F.type.b25846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.b51: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %F.type.b25846.2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41931.2: %F.type.b25846.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.b51 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc4: %F.type.b25846.1 = fn_decl @F.1 [concrete = constants.%F.c41931.1] {
// CHECK:STDOUT:     %T.patt.loc4_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc4_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %return.patt: @F.1.%ptr.loc4_30.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%ptr.loc4_30.2 (%ptr.79f) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc4_30.1: type = ptr_type %T [symbolic = %ptr.loc4_30.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc4_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc4_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_16.2 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.1.%ptr.loc4_30.2 (%ptr.79f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.1.%ptr.loc4_30.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc13: %F.type.b25846.2 = fn_decl @F.2 [concrete = constants.%F.c41931.2] {
// CHECK:STDOUT:     %T.patt.loc13_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc13_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc13_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %return.patt: @F.2.%ptr.loc13_30.2 (%ptr.b51) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.2.%ptr.loc13_30.2 (%ptr.b51) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc13_30.1: type = ptr_type %U [symbolic = %ptr.loc13_30.2 (constants.%ptr.b51)]
// CHECK:STDOUT:     %T.loc13_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc13_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.2.%ptr.loc13_30.2 (%ptr.b51) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.2.%ptr.loc13_30.2 (%ptr.b51) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(%T.loc4_6.1: type, %U.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc4_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_16.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc4_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %ptr.loc4_30.2: type = ptr_type @F.1.%T.loc4_6.2 (%T) [symbolic = %ptr.loc4_30.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_6.1: type, %U.patt.loc4_16.1: type) -> @F.1.%ptr.loc4_30.2 (%ptr.79f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(%T.loc13_6.1: type, %U.loc13_16.1: type) {
// CHECK:STDOUT:   %T.loc13_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc13_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc13_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc13_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %ptr.loc13_30.2: type = ptr_type @F.2.%U.loc13_16.2 (%U) [symbolic = %ptr.loc13_30.2 (constants.%ptr.b51)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.2.%ptr.loc13_30.2 (%ptr.b51) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc13_6.1: type, %U.patt.loc13_16.1: type) -> @F.2.%ptr.loc13_30.2 (%ptr.b51) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.b25846.1 = name_ref F, file.%F.decl.loc4 [concrete = constants.%F.c41931.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_6.2 => constants.%T.patt
// CHECK:STDOUT:   %U.loc4_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc4_16.2 => constants.%U.patt
// CHECK:STDOUT:   %ptr.loc4_30.2 => constants.%ptr.79f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%T.patt
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc13_16.2 => constants.%U.patt
// CHECK:STDOUT:   %ptr.loc13_30.2 => constants.%ptr.b51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_reorder.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U.336: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.7a9: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %F.type.b25846.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41931.1: %F.type.b25846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.e01: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %T.336: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.7a9: type = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %ptr.b51: type = ptr_type %T.336 [symbolic]
// CHECK:STDOUT:   %F.type.b25846.2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41931.2: %F.type.b25846.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.b51 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc4: %F.type.b25846.1 = fn_decl @F.1 [concrete = constants.%F.c41931.1] {
// CHECK:STDOUT:     %T.patt.loc4_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %U.patt.loc4_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_16.2 (constants.%U.patt.7a9)]
// CHECK:STDOUT:     %return.patt: @F.1.%ptr.loc4_30.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%ptr.loc4_30.2 (%ptr.79f) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %ptr.loc4_30.1: type = ptr_type %T.8b3 [symbolic = %ptr.loc4_30.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc4_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %U.loc4_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_16.2 (constants.%U.336)]
// CHECK:STDOUT:     %return.param: ref @F.1.%ptr.loc4_30.2 (%ptr.79f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.1.%ptr.loc4_30.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc13: %F.type.b25846.2 = fn_decl @F.2 [concrete = constants.%F.c41931.2] {
// CHECK:STDOUT:     %U.patt.loc13_6.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc13_6.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %T.patt.loc13_16.1: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc13_16.2 (constants.%T.patt.7a9)]
// CHECK:STDOUT:     %return.patt: @F.2.%ptr.loc13_30.2 (%ptr.b51) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.2.%ptr.loc13_30.2 (%ptr.b51) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, %T.loc13_16.1 [symbolic = %T.loc13_16.2 (constants.%T.336)]
// CHECK:STDOUT:     %ptr.loc13_30.1: type = ptr_type %T.336 [symbolic = %ptr.loc13_30.2 (constants.%ptr.b51)]
// CHECK:STDOUT:     %U.loc13_6.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc13_6.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %T.loc13_16.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc13_16.2 (constants.%T.336)]
// CHECK:STDOUT:     %return.param: ref @F.2.%ptr.loc13_30.2 (%ptr.b51) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.2.%ptr.loc13_30.2 (%ptr.b51) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(%T.loc4_6.1: type, %U.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc4_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %U.loc4_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_16.2 (constants.%U.336)]
// CHECK:STDOUT:   %U.patt.loc4_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_16.2 (constants.%U.patt.7a9)]
// CHECK:STDOUT:   %ptr.loc4_30.2: type = ptr_type @F.1.%T.loc4_6.2 (%T.8b3) [symbolic = %ptr.loc4_30.2 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_6.1: type, %U.patt.loc4_16.1: type) -> @F.1.%ptr.loc4_30.2 (%ptr.79f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(%U.loc13_6.1: type, %T.loc13_16.1: type) {
// CHECK:STDOUT:   %U.loc13_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc13_6.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %U.patt.loc13_6.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc13_6.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   %T.loc13_16.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc13_16.2 (constants.%T.336)]
// CHECK:STDOUT:   %T.patt.loc13_16.2: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc13_16.2 (constants.%T.patt.7a9)]
// CHECK:STDOUT:   %ptr.loc13_30.2: type = ptr_type @F.2.%T.loc13_16.2 (%T.336) [symbolic = %ptr.loc13_30.2 (constants.%ptr.b51)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.2.%ptr.loc13_30.2 (%ptr.b51) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc13_6.1: type, %T.patt.loc13_16.1: type) -> @F.2.%ptr.loc13_30.2 (%ptr.b51) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.b25846.1 = name_ref F, file.%F.decl.loc4 [concrete = constants.%F.c41931.1]
// CHECK:STDOUT:     %T.ref.loc21: type = name_ref T, %T.loc13_16.1 [symbolic = %T.loc13_16.2 (constants.%T.336)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.8b3, constants.%U.336) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc4_6.2 => constants.%T.patt.e01
// CHECK:STDOUT:   %U.loc4_16.2 => constants.%U.336
// CHECK:STDOUT:   %U.patt.loc4_16.2 => constants.%U.patt.7a9
// CHECK:STDOUT:   %ptr.loc4_30.2 => constants.%ptr.79f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U.8b3, constants.%T.336) {
// CHECK:STDOUT:   %U.loc13_6.2 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc13_6.2 => constants.%U.patt.e01
// CHECK:STDOUT:   %T.loc13_16.2 => constants.%T.336
// CHECK:STDOUT:   %T.patt.loc13_16.2 => constants.%T.patt.7a9
// CHECK:STDOUT:   %ptr.loc13_30.2 => constants.%ptr.b51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rename.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %U.336: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt.7a9: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %ptr.79f131.1: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %F.type.b25846.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41931.1: %F.type.b25846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U.8b3: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt.e01: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %T.336: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %T.patt.7a9: type = symbolic_binding_pattern T, 1 [symbolic]
// CHECK:STDOUT:   %ptr.79f131.2: type = ptr_type %U.8b3 [symbolic]
// CHECK:STDOUT:   %F.type.b25846.2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41931.2: %F.type.b25846.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.79f131.2 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc4: %F.type.b25846.1 = fn_decl @F.1 [concrete = constants.%F.c41931.1] {
// CHECK:STDOUT:     %T.patt.loc4_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:     %U.patt.loc4_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_16.2 (constants.%U.patt.7a9)]
// CHECK:STDOUT:     %return.patt: @F.1.%ptr.loc4_30.2 (%ptr.79f131.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%ptr.loc4_30.2 (%ptr.79f131.1) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %ptr.loc4_30.1: type = ptr_type %T.8b3 [symbolic = %ptr.loc4_30.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:     %T.loc4_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %U.loc4_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_16.2 (constants.%U.336)]
// CHECK:STDOUT:     %return.param: ref @F.1.%ptr.loc4_30.2 (%ptr.79f131.1) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.1.%ptr.loc4_30.2 (%ptr.79f131.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc13: %F.type.b25846.2 = fn_decl @F.2 [concrete = constants.%F.c41931.2] {
// CHECK:STDOUT:     %U.patt.loc13_6.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc13_6.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:     %T.patt.loc13_16.1: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc13_16.2 (constants.%T.patt.7a9)]
// CHECK:STDOUT:     %return.patt: @F.2.%ptr.loc13_30.2 (%ptr.79f131.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.2.%ptr.loc13_30.2 (%ptr.79f131.2) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc13_6.1 [symbolic = %U.loc13_6.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %ptr.loc13_30.1: type = ptr_type %U.8b3 [symbolic = %ptr.loc13_30.2 (constants.%ptr.79f131.2)]
// CHECK:STDOUT:     %U.loc13_6.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc13_6.2 (constants.%U.8b3)]
// CHECK:STDOUT:     %T.loc13_16.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc13_16.2 (constants.%T.336)]
// CHECK:STDOUT:     %return.param: ref @F.2.%ptr.loc13_30.2 (%ptr.79f131.2) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.2.%ptr.loc13_30.2 (%ptr.79f131.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(%T.loc4_6.1: type, %U.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc4_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   %U.loc4_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_16.2 (constants.%U.336)]
// CHECK:STDOUT:   %U.patt.loc4_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_16.2 (constants.%U.patt.7a9)]
// CHECK:STDOUT:   %ptr.loc4_30.2: type = ptr_type @F.1.%T.loc4_6.2 (%T.8b3) [symbolic = %ptr.loc4_30.2 (constants.%ptr.79f131.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_6.1: type, %U.patt.loc4_16.1: type) -> @F.1.%ptr.loc4_30.2 (%ptr.79f131.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(%U.loc13_6.1: type, %T.loc13_16.1: type) {
// CHECK:STDOUT:   %U.loc13_6.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc13_6.2 (constants.%U.8b3)]
// CHECK:STDOUT:   %U.patt.loc13_6.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc13_6.2 (constants.%U.patt.e01)]
// CHECK:STDOUT:   %T.loc13_16.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc13_16.2 (constants.%T.336)]
// CHECK:STDOUT:   %T.patt.loc13_16.2: type = symbolic_binding_pattern T, 1 [symbolic = %T.patt.loc13_16.2 (constants.%T.patt.7a9)]
// CHECK:STDOUT:   %ptr.loc13_30.2: type = ptr_type @F.2.%U.loc13_6.2 (%U.8b3) [symbolic = %ptr.loc13_30.2 (constants.%ptr.79f131.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.2.%ptr.loc13_30.2 (%ptr.79f131.2) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.patt.loc13_6.1: type, %T.patt.loc13_16.1: type) -> @F.2.%ptr.loc13_30.2 (%ptr.79f131.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.b25846.1 = name_ref F, file.%F.decl.loc4 [concrete = constants.%F.c41931.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13_16.1 [symbolic = %T.loc13_16.2 (constants.%T.336)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T.8b3, constants.%U.336) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc4_6.2 => constants.%T.patt.e01
// CHECK:STDOUT:   %U.loc4_16.2 => constants.%U.336
// CHECK:STDOUT:   %U.patt.loc4_16.2 => constants.%U.patt.7a9
// CHECK:STDOUT:   %ptr.loc4_30.2 => constants.%ptr.79f131.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U.8b3, constants.%T.336) {
// CHECK:STDOUT:   %U.loc13_6.2 => constants.%U.8b3
// CHECK:STDOUT:   %U.patt.loc13_6.2 => constants.%U.patt.e01
// CHECK:STDOUT:   %T.loc13_16.2 => constants.%T.336
// CHECK:STDOUT:   %T.patt.loc13_16.2 => constants.%T.patt.7a9
// CHECK:STDOUT:   %ptr.loc13_30.2 => constants.%ptr.79f131.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
