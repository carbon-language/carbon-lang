// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

// CHECK:STDERR: fail_reorder.carbon:[[@LINE+7]]:13: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fn F(unused U:! type, T:! type) -> T* {
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_reorder.carbon:[[@LINE-5]]:6: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: fn F(T:! type, U:! type) -> T*;
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(unused U:! type, T:! type) -> T* {
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc14: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc24 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc12: type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc13.2: type = ptr_type %T.ref.loc12 [symbolic = %ptr.loc13.1 (constants.%ptr)]
// CHECK:STDOUT:     %.loc13.2: Core.Form = init_form %ptr.loc13.2 [symbolic = %.loc13.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:     %return.param.loc13: ref @F.%ptr.loc13.1 (%ptr) = out_param call_param0
// CHECK:STDOUT:     %return.loc13: ref @F.%ptr.loc13.1 (%ptr) = return_slot %return.param.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc25: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc24 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc23: type = name_ref T, %T.loc18 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc24: type = ptr_type %T.ref.loc23 [symbolic = %ptr.loc13.1 (constants.%ptr)]
// CHECK:STDOUT:     %.loc24: Core.Form = init_form %ptr.loc24 [symbolic = %.loc13.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc20.1: type = splice_block %.loc20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:     %return.param.loc24: ref @F.%ptr.loc13.1 (%ptr) = out_param call_param0
// CHECK:STDOUT:     %return.loc24: ref @F.%ptr.loc13.1 (%ptr) = return_slot %return.param.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc13.1: type = ptr_type %T.loc7.1 [symbolic = %ptr.loc13.1 (constants.%ptr)]
// CHECK:STDOUT:   %.loc13.1: Core.Form = init_form %ptr.loc13.1 [symbolic = %.loc13.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc13.1 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc13.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc27.2: <specific function> = specific_function constants.%F, @F(%T.loc7.1) [symbolic = %F.specific_fn.loc27.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param.loc24: @F.%ptr.loc13.1 (%ptr) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl.loc14 [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc29: type = name_ref T, %T.loc18 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:     %F.specific_fn.loc27.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc27.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init @F.%ptr.loc13.1 (%ptr) = call %F.specific_fn.loc27.1()
// CHECK:STDOUT:     return %F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc13.1 => constants.%ptr
// CHECK:STDOUT:   %.loc13.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc27.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_different_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.1: type = fn_type @F.loc18 [concrete]
// CHECK:STDOUT:   %F.d98bd5.1: %F.type.117dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.18e: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %.6d8: Core.Form = init_form %ptr.18e [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.2: type = fn_type @F.loc33 [concrete]
// CHECK:STDOUT:   %F.d98bd5.2: %F.type.117dbc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.18e [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc18: %F.type.117dbc.1 = fn_decl @F.loc18 [concrete = constants.%F.d98bd5.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc18.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc18.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc17.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc17.2: type = ptr_type %T.ref [symbolic = %ptr.loc17.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc17.2: Core.Form = init_form %ptr.loc17.2 [symbolic = %.loc17.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11.2: type = symbolic_binding U, 1 [symbolic = %U.loc11.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.loc18.%ptr.loc17.1 (%ptr.e8f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc18.%ptr.loc17.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc33: %F.type.117dbc.2 = fn_decl @F.loc33 [concrete = constants.%F.d98bd5.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc33.%pattern_type (%pattern_type.423) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc33.%pattern_type (%pattern_type.423) = return_slot_pattern %return.param_patt, %ptr.loc32.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc26.2 [symbolic = %U.loc26.1 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc32.2: type = ptr_type %U.ref [symbolic = %ptr.loc32.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %.loc32.2: Core.Form = init_form %ptr.loc32.2 [symbolic = %.loc32.1 (constants.%.6d8)]
// CHECK:STDOUT:     %.loc24.1: type = splice_block %.loc24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22.2: type = symbolic_binding T, 0 [symbolic = %T.loc22.1 (constants.%T)]
// CHECK:STDOUT:     %.loc28.1: type = splice_block %.loc28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc26.2: type = symbolic_binding U, 1 [symbolic = %U.loc26.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.loc33.%ptr.loc32.1 (%ptr.18e) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc33.%ptr.loc32.1 (%ptr.18e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc18(%T.loc7.2: type, %U.loc11.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc11.1: type = symbolic_binding U, 1 [symbolic = %U.loc11.1 (constants.%U)]
// CHECK:STDOUT:   %ptr.loc17.1: type = ptr_type %T.loc7.1 [symbolic = %ptr.loc17.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.loc17.1: Core.Form = init_form %ptr.loc17.1 [symbolic = %.loc17.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc17.1 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc18.%ptr.loc17.1 (%ptr.e8f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc33(%T.loc22.2: type, %U.loc26.2: type) {
// CHECK:STDOUT:   %T.loc22.1: type = symbolic_binding T, 0 [symbolic = %T.loc22.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc26.1: type = symbolic_binding U, 1 [symbolic = %U.loc26.1 (constants.%U)]
// CHECK:STDOUT:   %ptr.loc32.1: type = ptr_type %U.loc26.1 [symbolic = %ptr.loc32.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %.loc32.1: Core.Form = init_form %ptr.loc32.1 [symbolic = %.loc32.1 (constants.%.6d8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc32.1 [symbolic = %pattern_type (constants.%pattern_type.423)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc32.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc33.%ptr.loc32.1 (%ptr.18e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.117dbc.1 = name_ref F, file.%F.decl.loc18 [concrete = constants.%F.d98bd5.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc22.2 [symbolic = %T.loc22.1 (constants.%T)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc18(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T
// CHECK:STDOUT:   %U.loc11.1 => constants.%U
// CHECK:STDOUT:   %ptr.loc17.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.loc17.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc33(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc22.1 => constants.%T
// CHECK:STDOUT:   %U.loc26.1 => constants.%U
// CHECK:STDOUT:   %ptr.loc32.1 => constants.%ptr.18e
// CHECK:STDOUT:   %.loc32.1 => constants.%.6d8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_reorder.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.1: type = fn_type @F.loc18 [concrete]
// CHECK:STDOUT:   %F.d98bd5.1: %F.type.117dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ptr.18e: type = ptr_type %T.091 [symbolic]
// CHECK:STDOUT:   %.6d8: Core.Form = init_form %ptr.18e [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.2: type = fn_type @F.loc34 [concrete]
// CHECK:STDOUT:   %F.d98bd5.2: %F.type.117dbc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.18e [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc18: %F.type.117dbc.1 = fn_decl @F.loc18 [concrete = constants.%F.d98bd5.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc18.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc18.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc17.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc17.2: type = ptr_type %T.ref [symbolic = %ptr.loc17.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc17.2: Core.Form = init_form %ptr.loc17.2 [symbolic = %.loc17.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11.2: type = symbolic_binding U, 1 [symbolic = %U.loc11.1 (constants.%U.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc18.%ptr.loc17.1 (%ptr.e8f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc18.%ptr.loc17.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc34: %F.type.117dbc.2 = fn_decl @F.loc34 [concrete = constants.%F.d98bd5.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc34.%pattern_type (%pattern_type.423) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc34.%pattern_type (%pattern_type.423) = return_slot_pattern %return.param_patt, %ptr.loc33.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc32: type = name_ref T, %T.loc27.2 [symbolic = %T.loc27.1 (constants.%T.091)]
// CHECK:STDOUT:     %ptr.loc33.2: type = ptr_type %T.ref.loc32 [symbolic = %ptr.loc33.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %.loc33.2: Core.Form = init_form %ptr.loc33.2 [symbolic = %.loc33.1 (constants.%.6d8)]
// CHECK:STDOUT:     %.loc25.1: type = splice_block %.loc25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc23.2: type = symbolic_binding U, 0 [symbolic = %U.loc23.1 (constants.%U.67d)]
// CHECK:STDOUT:     %.loc29.1: type = splice_block %.loc29.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc29.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc27.2: type = symbolic_binding T, 1 [symbolic = %T.loc27.1 (constants.%T.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc34.%ptr.loc33.1 (%ptr.18e) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc34.%ptr.loc33.1 (%ptr.18e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc18(%T.loc7.2: type, %U.loc11.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:   %U.loc11.1: type = symbolic_binding U, 1 [symbolic = %U.loc11.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc17.1: type = ptr_type %T.loc7.1 [symbolic = %ptr.loc17.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.loc17.1: Core.Form = init_form %ptr.loc17.1 [symbolic = %.loc17.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc17.1 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc18.%ptr.loc17.1 (%ptr.e8f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc34(%U.loc23.2: type, %T.loc27.2: type) {
// CHECK:STDOUT:   %U.loc23.1: type = symbolic_binding U, 0 [symbolic = %U.loc23.1 (constants.%U.67d)]
// CHECK:STDOUT:   %T.loc27.1: type = symbolic_binding T, 1 [symbolic = %T.loc27.1 (constants.%T.091)]
// CHECK:STDOUT:   %ptr.loc33.1: type = ptr_type %T.loc27.1 [symbolic = %ptr.loc33.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %.loc33.1: Core.Form = init_form %ptr.loc33.1 [symbolic = %.loc33.1 (constants.%.6d8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc33.1 [symbolic = %pattern_type (constants.%pattern_type.423)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc33.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc34.%ptr.loc33.1 (%ptr.18e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.117dbc.1 = name_ref F, file.%F.decl.loc18 [concrete = constants.%F.d98bd5.1]
// CHECK:STDOUT:     %T.ref.loc38: type = name_ref T, %T.loc27.2 [symbolic = %T.loc27.1 (constants.%T.091)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc18(constants.%T.67d, constants.%U.091) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.67d
// CHECK:STDOUT:   %U.loc11.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc17.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.loc17.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc34(constants.%U.67d, constants.%T.091) {
// CHECK:STDOUT:   %U.loc23.1 => constants.%U.67d
// CHECK:STDOUT:   %T.loc27.1 => constants.%T.091
// CHECK:STDOUT:   %ptr.loc33.1 => constants.%ptr.18e
// CHECK:STDOUT:   %.loc33.1 => constants.%.6d8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_rename.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.1: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %.cb6cb9.1: Core.Form = init_form %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.1: type = pattern_type %ptr.e8f8f9.1 [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.1: type = fn_type @F.loc18 [concrete]
// CHECK:STDOUT:   %F.d98bd5.1: %F.type.117dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %.cb6cb9.2: Core.Form = init_form %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.2: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.2: type = fn_type @F.loc33 [concrete]
// CHECK:STDOUT:   %F.d98bd5.2: %F.type.117dbc.2 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.e8f8f9.2 [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc18: %F.type.117dbc.1 = fn_decl @F.loc18 [concrete = constants.%F.d98bd5.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc18.%pattern_type (%pattern_type.4f4b84.1) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc18.%pattern_type (%pattern_type.4f4b84.1) = return_slot_pattern %return.param_patt, %ptr.loc17.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc17.2: type = ptr_type %T.ref [symbolic = %ptr.loc17.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc17.2: Core.Form = init_form %ptr.loc17.2 [symbolic = %.loc17.1 (constants.%.cb6cb9.1)]
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc13.1: type = splice_block %.loc13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc11.2: type = symbolic_binding U, 1 [symbolic = %U.loc11.1 (constants.%U.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc18.%ptr.loc17.1 (%ptr.e8f8f9.1) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc18.%ptr.loc17.1 (%ptr.e8f8f9.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc33: %F.type.117dbc.2 = fn_decl @F.loc33 [concrete = constants.%F.d98bd5.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc33.%pattern_type (%pattern_type.4f4b84.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc33.%pattern_type (%pattern_type.4f4b84.2) = return_slot_pattern %return.param_patt, %ptr.loc32.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc22.2 [symbolic = %U.loc22.1 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc32.2: type = ptr_type %U.ref [symbolic = %ptr.loc32.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %.loc32.2: Core.Form = init_form %ptr.loc32.2 [symbolic = %.loc32.1 (constants.%.cb6cb9.2)]
// CHECK:STDOUT:     %.loc24.1: type = splice_block %.loc24.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc24.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc22.2: type = symbolic_binding U, 0 [symbolic = %U.loc22.1 (constants.%U.67d)]
// CHECK:STDOUT:     %.loc28.1: type = splice_block %.loc28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26.2: type = symbolic_binding T, 1 [symbolic = %T.loc26.1 (constants.%T.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc33.%ptr.loc32.1 (%ptr.e8f8f9.2) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc33.%ptr.loc32.1 (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc18(%T.loc7.2: type, %U.loc11.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:   %U.loc11.1: type = symbolic_binding U, 1 [symbolic = %U.loc11.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc17.1: type = ptr_type %T.loc7.1 [symbolic = %ptr.loc17.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %.loc17.1: Core.Form = init_form %ptr.loc17.1 [symbolic = %.loc17.1 (constants.%.cb6cb9.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc17.1 [symbolic = %pattern_type (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc18.%ptr.loc17.1 (%ptr.e8f8f9.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc33(%U.loc22.2: type, %T.loc26.2: type) {
// CHECK:STDOUT:   %U.loc22.1: type = symbolic_binding U, 0 [symbolic = %U.loc22.1 (constants.%U.67d)]
// CHECK:STDOUT:   %T.loc26.1: type = symbolic_binding T, 1 [symbolic = %T.loc26.1 (constants.%T.091)]
// CHECK:STDOUT:   %ptr.loc32.1: type = ptr_type %U.loc22.1 [symbolic = %ptr.loc32.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc32.1: Core.Form = init_form %ptr.loc32.1 [symbolic = %.loc32.1 (constants.%.cb6cb9.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc32.1 [symbolic = %pattern_type (constants.%pattern_type.4f4b84.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc32.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc33.%ptr.loc32.1 (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.117dbc.1 = name_ref F, file.%F.decl.loc18 [concrete = constants.%F.d98bd5.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc26.2 [symbolic = %T.loc26.1 (constants.%T.091)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc18(constants.%T.67d, constants.%U.091) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.67d
// CHECK:STDOUT:   %U.loc11.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc17.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %.loc17.1 => constants.%.cb6cb9.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc33(constants.%U.67d, constants.%T.091) {
// CHECK:STDOUT:   %U.loc22.1 => constants.%U.67d
// CHECK:STDOUT:   %T.loc26.1 => constants.%T.091
// CHECK:STDOUT:   %ptr.loc32.1 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc32.1 => constants.%.cb6cb9.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4b84.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
