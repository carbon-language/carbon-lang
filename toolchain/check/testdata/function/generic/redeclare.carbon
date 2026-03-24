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
// CHECK:STDOUT:     .F = %F.decl.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc19: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc21 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19: type = name_ref T, %T.loc19_6.2 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc19_20.2: type = ptr_type %T.ref.loc19 [symbolic = %ptr.loc19_20.1 (constants.%ptr)]
// CHECK:STDOUT:     %.loc19_20.2: Core.Form = init_form %ptr.loc19_20.2 [symbolic = %.loc19_20.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc19_10.1: type = splice_block %.loc19_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc19_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %return.param.loc19: ref @F.%ptr.loc19_20.1 (%ptr) = out_param call_param0
// CHECK:STDOUT:     %return.loc19: ref @F.%ptr.loc19_20.1 (%ptr) = return_slot %return.param.loc19
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc21: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc21 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc21: type = name_ref T, %T.loc21 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc21: type = ptr_type %T.ref.loc21 [symbolic = %ptr.loc19_20.1 (constants.%ptr)]
// CHECK:STDOUT:     %.loc21_20: Core.Form = init_form %ptr.loc21 [symbolic = %.loc19_20.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc21_10.1: type = splice_block %.loc21_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc21_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21: type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %return.param.loc21: ref @F.%ptr.loc19_20.1 (%ptr) = out_param call_param0
// CHECK:STDOUT:     %return.loc21: ref @F.%ptr.loc19_20.1 (%ptr) = return_slot %return.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc19_6.2: type) {
// CHECK:STDOUT:   %T.loc19_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc19_20.1: type = ptr_type %T.loc19_6.1 [symbolic = %ptr.loc19_20.1 (constants.%ptr)]
// CHECK:STDOUT:   %.loc19_20.1: Core.Form = init_form %ptr.loc19_20.1 [symbolic = %.loc19_20.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc19_20.1 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc19_20.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc22_10.2: <specific function> = specific_function constants.%F, @F(%T.loc19_6.1) [symbolic = %F.specific_fn.loc22_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param.loc21: @F.%ptr.loc19_20.1 (%ptr) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl.loc19 [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc22: type = name_ref T, %T.loc21 [symbolic = %T.loc19_6.1 (constants.%T)]
// CHECK:STDOUT:     %F.specific_fn.loc22_10.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc22_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init @F.%ptr.loc19_20.1 (%ptr) = call %F.specific_fn.loc22_10.1()
// CHECK:STDOUT:     return %F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc19_20.1 => constants.%ptr
// CHECK:STDOUT:   %.loc19_20.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc22_10.2 => constants.%F.specific_fn
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
// CHECK:STDOUT:   %F.type.117dbc.1: type = fn_type @F.loc29 [concrete]
// CHECK:STDOUT:   %F.d98bd5.1: %F.type.117dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.18e: type = ptr_type %U [symbolic]
// CHECK:STDOUT:   %.6d8: Core.Form = init_form %ptr.18e [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.2: type = fn_type @F.loc38 [concrete]
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
// CHECK:STDOUT:     .F = %F.decl.loc29
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc29: %F.type.117dbc.1 = fn_decl @F.loc29 [concrete = constants.%F.d98bd5.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc29.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc29.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc29_30.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc29_6.2 [symbolic = %T.loc29_6.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc29_30.2: type = ptr_type %T.ref [symbolic = %ptr.loc29_30.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc29_30.2: Core.Form = init_form %ptr.loc29_30.2 [symbolic = %.loc29_30.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc29_10.1: type = splice_block %.loc29_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc29_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc29_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc29_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc29_20.1: type = splice_block %.loc29_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc29_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc29_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc29_16.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.loc29.%ptr.loc29_30.1 (%ptr.e8f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc29.%ptr.loc29_30.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc38: %F.type.117dbc.2 = fn_decl @F.loc38 [concrete = constants.%F.d98bd5.2] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc38.%pattern_type (%pattern_type.423) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc38.%pattern_type (%pattern_type.423) = return_slot_pattern %return.param_patt, %ptr.loc38_30.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc38_16.2 [symbolic = %U.loc38_16.1 (constants.%U)]
// CHECK:STDOUT:     %ptr.loc38_30.2: type = ptr_type %U.ref [symbolic = %ptr.loc38_30.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %.loc38_30.2: Core.Form = init_form %ptr.loc38_30.2 [symbolic = %.loc38_30.1 (constants.%.6d8)]
// CHECK:STDOUT:     %.loc38_10.1: type = splice_block %.loc38_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc38_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc38_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc38_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc38_20.1: type = splice_block %.loc38_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc38_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc38_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc38_16.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.loc38.%ptr.loc38_30.1 (%ptr.18e) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc38.%ptr.loc38_30.1 (%ptr.18e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc29(%T.loc29_6.2: type, %U.loc29_16.2: type) {
// CHECK:STDOUT:   %T.loc29_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc29_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc29_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc29_16.1 (constants.%U)]
// CHECK:STDOUT:   %ptr.loc29_30.1: type = ptr_type %T.loc29_6.1 [symbolic = %ptr.loc29_30.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.loc29_30.1: Core.Form = init_form %ptr.loc29_30.1 [symbolic = %.loc29_30.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc29_30.1 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc29.%ptr.loc29_30.1 (%ptr.e8f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc38(%T.loc38_6.2: type, %U.loc38_16.2: type) {
// CHECK:STDOUT:   %T.loc38_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc38_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc38_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc38_16.1 (constants.%U)]
// CHECK:STDOUT:   %ptr.loc38_30.1: type = ptr_type %U.loc38_16.1 [symbolic = %ptr.loc38_30.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %.loc38_30.1: Core.Form = init_form %ptr.loc38_30.1 [symbolic = %.loc38_30.1 (constants.%.6d8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc38_30.1 [symbolic = %pattern_type (constants.%pattern_type.423)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc38_30.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc38.%ptr.loc38_30.1 (%ptr.18e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.117dbc.1 = name_ref F, file.%F.decl.loc29 [concrete = constants.%F.d98bd5.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc38_6.2 [symbolic = %T.loc38_6.1 (constants.%T)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc29(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc29_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc29_16.1 => constants.%U
// CHECK:STDOUT:   %ptr.loc29_30.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.loc29_30.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc38(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc38_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc38_16.1 => constants.%U
// CHECK:STDOUT:   %ptr.loc38_30.1 => constants.%ptr.18e
// CHECK:STDOUT:   %.loc38_30.1 => constants.%.6d8
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
// CHECK:STDOUT:   %F.type.117dbc.1: type = fn_type @F.loc53 [concrete]
// CHECK:STDOUT:   %F.d98bd5.1: %F.type.117dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ptr.18e: type = ptr_type %T.091 [symbolic]
// CHECK:STDOUT:   %.6d8: Core.Form = init_form %ptr.18e [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %ptr.18e [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.2: type = fn_type @F.loc62 [concrete]
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
// CHECK:STDOUT:     .F = %F.decl.loc53
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc53: %F.type.117dbc.1 = fn_decl @F.loc53 [concrete = constants.%F.d98bd5.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc53.%pattern_type (%pattern_type.4f4) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc53.%pattern_type (%pattern_type.4f4) = return_slot_pattern %return.param_patt, %ptr.loc53_30.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc53_6.2 [symbolic = %T.loc53_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc53_30.2: type = ptr_type %T.ref [symbolic = %ptr.loc53_30.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %.loc53_30.2: Core.Form = init_form %ptr.loc53_30.2 [symbolic = %.loc53_30.1 (constants.%.cb6)]
// CHECK:STDOUT:     %.loc53_10.1: type = splice_block %.loc53_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc53_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc53_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc53_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc53_20.1: type = splice_block %.loc53_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc53_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc53_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc53_16.1 (constants.%U.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc53.%ptr.loc53_30.1 (%ptr.e8f) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc53.%ptr.loc53_30.1 (%ptr.e8f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc62: %F.type.117dbc.2 = fn_decl @F.loc62 [concrete = constants.%F.d98bd5.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc62.%pattern_type (%pattern_type.423) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc62.%pattern_type (%pattern_type.423) = return_slot_pattern %return.param_patt, %ptr.loc62_37.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc62: type = name_ref T, %T.loc62_23.2 [symbolic = %T.loc62_23.1 (constants.%T.091)]
// CHECK:STDOUT:     %ptr.loc62_37.2: type = ptr_type %T.ref.loc62 [symbolic = %ptr.loc62_37.1 (constants.%ptr.18e)]
// CHECK:STDOUT:     %.loc62_37.2: Core.Form = init_form %ptr.loc62_37.2 [symbolic = %.loc62_37.1 (constants.%.6d8)]
// CHECK:STDOUT:     %.loc62_17.1: type = splice_block %.loc62_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc62_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc62_13.2: type = symbolic_binding U, 0 [symbolic = %U.loc62_13.1 (constants.%U.67d)]
// CHECK:STDOUT:     %.loc62_27.1: type = splice_block %.loc62_27.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc62_27.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc62_23.2: type = symbolic_binding T, 1 [symbolic = %T.loc62_23.1 (constants.%T.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc62.%ptr.loc62_37.1 (%ptr.18e) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc62.%ptr.loc62_37.1 (%ptr.18e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc53(%T.loc53_6.2: type, %U.loc53_16.2: type) {
// CHECK:STDOUT:   %T.loc53_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc53_6.1 (constants.%T.67d)]
// CHECK:STDOUT:   %U.loc53_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc53_16.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc53_30.1: type = ptr_type %T.loc53_6.1 [symbolic = %ptr.loc53_30.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %.loc53_30.1: Core.Form = init_form %ptr.loc53_30.1 [symbolic = %.loc53_30.1 (constants.%.cb6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc53_30.1 [symbolic = %pattern_type (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc53.%ptr.loc53_30.1 (%ptr.e8f);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc62(%U.loc62_13.2: type, %T.loc62_23.2: type) {
// CHECK:STDOUT:   %U.loc62_13.1: type = symbolic_binding U, 0 [symbolic = %U.loc62_13.1 (constants.%U.67d)]
// CHECK:STDOUT:   %T.loc62_23.1: type = symbolic_binding T, 1 [symbolic = %T.loc62_23.1 (constants.%T.091)]
// CHECK:STDOUT:   %ptr.loc62_37.1: type = ptr_type %T.loc62_23.1 [symbolic = %ptr.loc62_37.1 (constants.%ptr.18e)]
// CHECK:STDOUT:   %.loc62_37.1: Core.Form = init_form %ptr.loc62_37.1 [symbolic = %.loc62_37.1 (constants.%.6d8)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc62_37.1 [symbolic = %pattern_type (constants.%pattern_type.423)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc62_37.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc62.%ptr.loc62_37.1 (%ptr.18e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.117dbc.1 = name_ref F, file.%F.decl.loc53 [concrete = constants.%F.d98bd5.1]
// CHECK:STDOUT:     %T.ref.loc70: type = name_ref T, %T.loc62_23.2 [symbolic = %T.loc62_23.1 (constants.%T.091)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc53(constants.%T.67d, constants.%U.091) {
// CHECK:STDOUT:   %T.loc53_6.1 => constants.%T.67d
// CHECK:STDOUT:   %U.loc53_16.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc53_30.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %.loc53_30.1 => constants.%.cb6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc62(constants.%U.67d, constants.%T.091) {
// CHECK:STDOUT:   %U.loc62_13.1 => constants.%U.67d
// CHECK:STDOUT:   %T.loc62_23.1 => constants.%T.091
// CHECK:STDOUT:   %ptr.loc62_37.1 => constants.%ptr.18e
// CHECK:STDOUT:   %.loc62_37.1 => constants.%.6d8
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
// CHECK:STDOUT:   %F.type.117dbc.1: type = fn_type @F.loc77 [concrete]
// CHECK:STDOUT:   %F.d98bd5.1: %F.type.117dbc.1 = struct_value () [concrete]
// CHECK:STDOUT:   %U.67d: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %ptr.e8f8f9.2: type = ptr_type %U.67d [symbolic]
// CHECK:STDOUT:   %.cb6cb9.2: Core.Form = init_form %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4b84.2: type = pattern_type %ptr.e8f8f9.2 [symbolic]
// CHECK:STDOUT:   %F.type.117dbc.2: type = fn_type @F.loc86 [concrete]
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
// CHECK:STDOUT:     .F = %F.decl.loc77
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl.loc77: %F.type.117dbc.1 = fn_decl @F.loc77 [concrete = constants.%F.d98bd5.1] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc77.%pattern_type (%pattern_type.4f4b84.1) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc77.%pattern_type (%pattern_type.4f4b84.1) = return_slot_pattern %return.param_patt, %ptr.loc77_30.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc77_6.2 [symbolic = %T.loc77_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc77_30.2: type = ptr_type %T.ref [symbolic = %ptr.loc77_30.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:     %.loc77_30.2: Core.Form = init_form %ptr.loc77_30.2 [symbolic = %.loc77_30.1 (constants.%.cb6cb9.1)]
// CHECK:STDOUT:     %.loc77_10.1: type = splice_block %.loc77_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc77_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc77_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc77_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc77_20.1: type = splice_block %.loc77_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc77_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc77_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc77_16.1 (constants.%U.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc77.%ptr.loc77_30.1 (%ptr.e8f8f9.1) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc77.%ptr.loc77_30.1 (%ptr.e8f8f9.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc86: %F.type.117dbc.2 = fn_decl @F.loc86 [concrete = constants.%F.d98bd5.2] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.loc86.%pattern_type (%pattern_type.4f4b84.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.loc86.%pattern_type (%pattern_type.4f4b84.2) = return_slot_pattern %return.param_patt, %ptr.loc86_30.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc86_6.2 [symbolic = %U.loc86_6.1 (constants.%U.67d)]
// CHECK:STDOUT:     %ptr.loc86_30.2: type = ptr_type %U.ref [symbolic = %ptr.loc86_30.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:     %.loc86_30.2: Core.Form = init_form %ptr.loc86_30.2 [symbolic = %.loc86_30.1 (constants.%.cb6cb9.2)]
// CHECK:STDOUT:     %.loc86_10.1: type = splice_block %.loc86_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc86_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc86_6.2: type = symbolic_binding U, 0 [symbolic = %U.loc86_6.1 (constants.%U.67d)]
// CHECK:STDOUT:     %.loc86_20.1: type = splice_block %.loc86_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc86_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc86_16.2: type = symbolic_binding T, 1 [symbolic = %T.loc86_16.1 (constants.%T.091)]
// CHECK:STDOUT:     %return.param: ref @F.loc86.%ptr.loc86_30.1 (%ptr.e8f8f9.2) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.loc86.%ptr.loc86_30.1 (%ptr.e8f8f9.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc77(%T.loc77_6.2: type, %U.loc77_16.2: type) {
// CHECK:STDOUT:   %T.loc77_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc77_6.1 (constants.%T.67d)]
// CHECK:STDOUT:   %U.loc77_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc77_16.1 (constants.%U.091)]
// CHECK:STDOUT:   %ptr.loc77_30.1: type = ptr_type %T.loc77_6.1 [symbolic = %ptr.loc77_30.1 (constants.%ptr.e8f8f9.1)]
// CHECK:STDOUT:   %.loc77_30.1: Core.Form = init_form %ptr.loc77_30.1 [symbolic = %.loc77_30.1 (constants.%.cb6cb9.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc77_30.1 [symbolic = %pattern_type (constants.%pattern_type.4f4b84.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc77.%ptr.loc77_30.1 (%ptr.e8f8f9.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc86(%U.loc86_6.2: type, %T.loc86_16.2: type) {
// CHECK:STDOUT:   %U.loc86_6.1: type = symbolic_binding U, 0 [symbolic = %U.loc86_6.1 (constants.%U.67d)]
// CHECK:STDOUT:   %T.loc86_16.1: type = symbolic_binding T, 1 [symbolic = %T.loc86_16.1 (constants.%T.091)]
// CHECK:STDOUT:   %ptr.loc86_30.1: type = ptr_type %U.loc86_6.1 [symbolic = %ptr.loc86_30.1 (constants.%ptr.e8f8f9.2)]
// CHECK:STDOUT:   %.loc86_30.1: Core.Form = init_form %ptr.loc86_30.1 [symbolic = %.loc86_30.1 (constants.%.cb6cb9.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc86_30.1 [symbolic = %pattern_type (constants.%pattern_type.4f4b84.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc86_30.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @F.loc86.%ptr.loc86_30.1 (%ptr.e8f8f9.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type.117dbc.1 = name_ref F, file.%F.decl.loc77 [concrete = constants.%F.d98bd5.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc86_16.2 [symbolic = %T.loc86_16.1 (constants.%T.091)]
// CHECK:STDOUT:     return <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc77(constants.%T.67d, constants.%U.091) {
// CHECK:STDOUT:   %T.loc77_6.1 => constants.%T.67d
// CHECK:STDOUT:   %U.loc77_16.1 => constants.%U.091
// CHECK:STDOUT:   %ptr.loc77_30.1 => constants.%ptr.e8f8f9.1
// CHECK:STDOUT:   %.loc77_30.1 => constants.%.cb6cb9.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4b84.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc86(constants.%U.67d, constants.%T.091) {
// CHECK:STDOUT:   %U.loc86_6.1 => constants.%U.67d
// CHECK:STDOUT:   %T.loc86_16.1 => constants.%T.091
// CHECK:STDOUT:   %ptr.loc86_30.1 => constants.%ptr.e8f8f9.2
// CHECK:STDOUT:   %.loc86_30.1 => constants.%.cb6cb9.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4b84.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
