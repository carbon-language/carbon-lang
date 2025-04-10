// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template_dependence.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template_dependence.carbon

// --- type.carbon

library "[[@TEST_NAME]]";

fn F[template T:! type](x: T**) -> T* {
  return *x;
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn F(template T:! type, U:! type) -> (T, U) {
  return F(T, U);
}

// CHECK:STDOUT: --- type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [template]
// CHECK:STDOUT:   %ptr.a13: type = ptr_type %ptr.79f [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [template]
// CHECK:STDOUT:   %require_complete.132: <witness> = require_complete_type %ptr.a13 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.%ptr.loc4_30.2 (%ptr.a13) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%ptr.loc4_30.2 (%ptr.a13) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%ptr.loc4_29.2 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%ptr.loc4_29.2 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_36: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc4_37: type = ptr_type %T.ref.loc4_36 [template = %ptr.loc4_29.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%ptr.loc4_30.2 (%ptr.a13) = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = splice_block %ptr.loc4_30.1 [template = %ptr.loc4_30.2 (constants.%ptr.a13)] {
// CHECK:STDOUT:       %T.ref.loc4_28: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc4_29.1: type = ptr_type %T.ref.loc4_28 [template = %ptr.loc4_29.2 (constants.%ptr.79f)]
// CHECK:STDOUT:       %ptr.loc4_30.1: type = ptr_type %ptr.loc4_29.1 [template = %ptr.loc4_30.2 (constants.%ptr.a13)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%ptr.loc4_30.2 (%ptr.a13) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc4_29.2 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc4_29.2 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %ptr.loc4_29.2: type = ptr_type %T.loc4_15.2 [template = %ptr.loc4_29.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %ptr.loc4_30.2: type = ptr_type %ptr.loc4_29.2 [template = %ptr.loc4_30.2 (constants.%ptr.a13)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_33: <witness> = require_complete_type @F.%ptr.loc4_29.2 (%ptr.79f) [template = %require_complete.loc4_33 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc4_26: <witness> = require_complete_type @F.%ptr.loc4_30.2 (%ptr.a13) [template = %require_complete.loc4_26 (constants.%require_complete.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_15.1: type](%x.param_patt: @F.%ptr.loc4_30.2 (%ptr.a13)) -> @F.%ptr.loc4_29.2 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%ptr.loc4_30.2 (%ptr.a13) = name_ref x, %x
// CHECK:STDOUT:     %.loc5_10.1: ref @F.%ptr.loc4_29.2 (%ptr.79f) = deref %x.ref
// CHECK:STDOUT:     %.loc5_10.2: @F.%ptr.loc4_29.2 (%ptr.79f) = bind_value %.loc5_10.1
// CHECK:STDOUT:     return %.loc5_10.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %ptr.loc4_29.2 => constants.%ptr.79f
// CHECK:STDOUT:   %ptr.loc4_30.2 => constants.%ptr.a13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mixed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0, template [template]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [template]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T, %U) [template]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc4_25.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_25.2 (constants.%U.patt)]
// CHECK:STDOUT:     %return.patt: @F.%tuple.type (%tuple.type.30b) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%tuple.type (%tuple.type.30b) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc4: type = name_ref U, %U.loc4_25.1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:     %.loc4_43.1: %tuple.type.24b = tuple_literal (%T.ref.loc4, %U.ref.loc4)
// CHECK:STDOUT:     %.loc4_43.2: type = converted %.loc4_43.1, constants.%tuple.type.30b [template = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc4_25.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.30b) = out_param call_param0
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_15.1: type, %U.loc4_25.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0, template [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc4_25.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc4_25.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc4_25.2 (constants.%U.patt)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%T.loc4_15.2 (%T), @F.%U.loc4_25.2 (%U)) [template = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4: <witness> = require_complete_type @F.%tuple.type (%tuple.type.30b) [template = %require_complete.loc4 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %F.specific_fn.loc5_10.2: <specific function> = specific_function constants.%F, @F(%T.loc4_15.2, %U.loc4_25.2) [template = %F.specific_fn.loc5_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:   %require_complete.loc5_16.1: <witness> = require_complete_type @F.%T.loc4_15.2 (%T) [template = %require_complete.loc5_16.1 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc5_16.2: <witness> = require_complete_type @F.%U.loc4_25.2 (%U) [symbolic = %require_complete.loc5_16.2 (constants.%require_complete.b54)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_15.1: type, %U.patt.loc4_25.1: type) -> %return.param_patt: @F.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc5: type = name_ref T, %T.loc4_15.1 [template = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc5: type = name_ref U, %U.loc4_25.1 [symbolic = %U.loc4_25.2 (constants.%U)]
// CHECK:STDOUT:     %F.specific_fn.loc5_10.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [template = %F.specific_fn.loc5_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %.loc5_16.1: ref @F.%tuple.type (%tuple.type.30b) = temporary_storage
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.30b) = call %F.specific_fn.loc5_10.1() to %.loc5_16.1
// CHECK:STDOUT:     %.loc5_16.2: ref @F.%tuple.type (%tuple.type.30b) = temporary %.loc5_16.1, %F.call
// CHECK:STDOUT:     %tuple.elem0.loc5_16.1: ref @F.%T.loc4_15.2 (%T) = tuple_access %.loc5_16.2, element0
// CHECK:STDOUT:     %.loc5_16.3: @F.%T.loc4_15.2 (%T) = bind_value %tuple.elem0.loc5_16.1
// CHECK:STDOUT:     %tuple.elem0.loc5_16.2: ref @F.%T.loc4_15.2 (%T) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc5_16.4: init @F.%T.loc4_15.2 (%T) = initialize_from %.loc5_16.3 to %tuple.elem0.loc5_16.2
// CHECK:STDOUT:     %tuple.elem1.loc5_16.1: ref @F.%U.loc4_25.2 (%U) = tuple_access %.loc5_16.2, element1
// CHECK:STDOUT:     %.loc5_16.5: @F.%U.loc4_25.2 (%U) = bind_value %tuple.elem1.loc5_16.1
// CHECK:STDOUT:     %tuple.elem1.loc5_16.2: ref @F.%U.loc4_25.2 (%U) = tuple_access %return, element1
// CHECK:STDOUT:     %.loc5_16.6: init @F.%U.loc4_25.2 (%U) = initialize_from %.loc5_16.5 to %tuple.elem1.loc5_16.2
// CHECK:STDOUT:     %.loc5_16.7: init @F.%tuple.type (%tuple.type.30b) = tuple_init (%.loc5_16.4, %.loc5_16.6) to %return
// CHECK:STDOUT:     %.loc5_17: init @F.%tuple.type (%tuple.type.30b) = converted %F.call, %.loc5_16.7
// CHECK:STDOUT:     return %.loc5_17 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.patt
// CHECK:STDOUT:   %U.loc4_25.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc4_25.2 => constants.%U.patt
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4 => constants.%require_complete.fe1
// CHECK:STDOUT:   %F.specific_fn.loc5_10.2 => constants.%F.specific_fn
// CHECK:STDOUT:   %require_complete.loc5_16.1 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc5_16.2 => constants.%require_complete.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc4_15.2, %U.loc4_25.2) {}
// CHECK:STDOUT:
