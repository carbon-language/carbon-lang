// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/min_prelude/fail_type_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/min_prelude/fail_type_param_mismatch.carbon

fn F(T:! type, U:! type) {
  var p: T*;
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+4]]:14: error: name `Core.ImplicitAs` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   let n: U = *p;
  // CHECK:STDERR:              ^~
  // CHECK:STDERR:
  let n: U = *p;
}

// CHECK:STDOUT: --- fail_type_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.a32: type = pattern_type %U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = <poisoned>
// CHECK:STDOUT:     import Core//prelude
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc13_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc13_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: type, %U.loc13_16.1: type) {
// CHECK:STDOUT:   %T.loc13_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %U.loc13_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc14_11.2: type = ptr_type %T.loc13_6.2 [symbolic = %ptr.loc14_11.2 (constants.%ptr)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %ptr.loc14_11.2 [symbolic = %require_complete.loc14 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %pattern_type.loc14: type = pattern_type %ptr.loc14_11.2 [symbolic = %pattern_type.loc14 (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %require_complete.loc19: <witness> = require_complete_type %U.loc13_16.2 [symbolic = %require_complete.loc19 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %pattern_type.loc19: type = pattern_type %U.loc13_16.2 [symbolic = %pattern_type.loc19 (constants.%pattern_type.a32)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %p.patt: @F.%pattern_type.loc14 (%pattern_type.afe) = binding_pattern p [concrete]
// CHECK:STDOUT:       %p.var_patt: @F.%pattern_type.loc14 (%pattern_type.afe) = var_pattern %p.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.var: ref @F.%ptr.loc14_11.2 (%ptr) = var %p.var_patt
// CHECK:STDOUT:     %.loc14: type = splice_block %ptr.loc14_11.1 [symbolic = %ptr.loc14_11.2 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc14_11.1: type = ptr_type %T.ref [symbolic = %ptr.loc14_11.2 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: ref @F.%ptr.loc14_11.2 (%ptr) = bind_name p, %p.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: @F.%pattern_type.loc19 (%pattern_type.a32) = binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.ref: ref @F.%ptr.loc14_11.2 (%ptr) = name_ref p, %p
// CHECK:STDOUT:     %.loc19_15: @F.%ptr.loc14_11.2 (%ptr) = bind_value %p.ref
// CHECK:STDOUT:     %.loc19_14.1: ref @F.%T.loc13_6.2 (%T) = deref %.loc19_15
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %.loc19_14.2: @F.%U.loc13_16.2 (%U) = converted %.loc19_14.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %n: @F.%U.loc13_16.2 (%U) = bind_name n, <error>
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
