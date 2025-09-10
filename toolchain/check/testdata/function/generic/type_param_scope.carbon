// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/type_param_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/type_param_scope.carbon

fn F(T:! type, n: T) -> T {
  let m: T = n;
  return m;
}

// CHECK:STDOUT: --- type_param_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %n.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: @F.%pattern_type (%pattern_type.7dc) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc15_25: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc15_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:     %n.param: @F.%T.loc15_6.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc15_19: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:     %n: @F.%T.loc15_6.1 (%T) = bind_name n, %n.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc15_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc15_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc15_6.2: type) {
// CHECK:STDOUT:   %T.loc15_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc15_6.1 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc15_6.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%T.loc15_6.1 (%T)) -> @F.%T.loc15_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %m.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern m [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.ref: @F.%T.loc15_6.1 (%T) = name_ref n, %n
// CHECK:STDOUT:     %T.ref.loc16: type = name_ref T, %T.loc15_6.2 [symbolic = %T.loc15_6.1 (constants.%T)]
// CHECK:STDOUT:     %m: @F.%T.loc15_6.1 (%T) = bind_name m, %n.ref
// CHECK:STDOUT:     %m.ref: @F.%T.loc15_6.1 (%T) = name_ref m, %m
// CHECK:STDOUT:     return %m.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc15_6.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
