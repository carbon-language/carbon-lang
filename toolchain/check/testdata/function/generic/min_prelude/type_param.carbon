// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
// EXTRA-ARGS: --custom-core
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/min_prelude/type_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/min_prelude/type_param.carbon

fn F(T:! type) {
  var p: T*;
  let n: T = *p;
}

// CHECK:STDOUT: --- type_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:     %T.patt.loc14_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc14_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc14_6.1: type) {
// CHECK:STDOUT:   %T.loc14_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc14_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc14_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc15_11.2: type = ptr_type %T.loc14_6.2 [symbolic = %ptr.loc15_11.2 (constants.%ptr)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %ptr.loc15_11.2 [symbolic = %require_complete.loc15 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %T.loc14_6.2 [symbolic = %require_complete.loc16 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %p.patt: @F.%ptr.loc15_11.2 (%ptr) = binding_pattern p
// CHECK:STDOUT:       %.loc15_3: @F.%ptr.loc15_11.2 (%ptr) = var_pattern %p.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.var: ref @F.%ptr.loc15_11.2 (%ptr) = var p
// CHECK:STDOUT:     %.loc15_11: type = splice_block %ptr.loc15_11.1 [symbolic = %ptr.loc15_11.2 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref.loc15: type = name_ref T, %T.loc14_6.1 [symbolic = %T.loc14_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc15_11.1: type = ptr_type %T.ref.loc15 [symbolic = %ptr.loc15_11.2 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: ref @F.%ptr.loc15_11.2 (%ptr) = bind_name p, %p.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: @F.%T.loc14_6.2 (%T) = binding_pattern n
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.ref: ref @F.%ptr.loc15_11.2 (%ptr) = name_ref p, %p
// CHECK:STDOUT:     %.loc16_15: @F.%ptr.loc15_11.2 (%ptr) = bind_value %p.ref
// CHECK:STDOUT:     %.loc16_14: ref @F.%T.loc14_6.2 (%T) = deref %.loc16_15
// CHECK:STDOUT:     %T.ref.loc16: type = name_ref T, %T.loc14_6.1 [symbolic = %T.loc14_6.2 (constants.%T)]
// CHECK:STDOUT:     %n: ref @F.%T.loc14_6.2 (%T) = bind_name n, %.loc16_14
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc14_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc14_6.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- include_files/destroy.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
