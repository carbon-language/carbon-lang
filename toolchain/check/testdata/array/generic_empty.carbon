// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/generic_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/generic_empty.carbon

fn G(T:! type) {
  // We can initialize this without knowing T.
  var arr: array(T, 0) = ();
}

fn H() {
  G(i32);
}

// CHECK:STDOUT: --- generic_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %array_type.281: type = array_type %int_0, %T [symbolic]
// CHECK:STDOUT:   %require_complete.b7f: <witness> = require_complete_type %array_type.281 [symbolic]
// CHECK:STDOUT:   %pattern_type.d48: type = pattern_type %array_type.281 [symbolic]
// CHECK:STDOUT:   %array.2ed: %array_type.281 = tuple_value () [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%i32) [concrete]
// CHECK:STDOUT:   %array_type.f5c: type = array_type %int_0, %i32 [concrete]
// CHECK:STDOUT:   %complete_type.cff: <witness> = complete_type_witness %array_type.f5c [concrete]
// CHECK:STDOUT:   %pattern_type.772: type = pattern_type %array_type.f5c [concrete]
// CHECK:STDOUT:   %array.812: %array_type.f5c = tuple_value () [concrete]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc11_6.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc11_6.1: type) {
// CHECK:STDOUT:   %T.loc11_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_6.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc13_22.2: type = array_type constants.%int_0, %T.loc11_6.2 [symbolic = %array_type.loc13_22.2 (constants.%array_type.281)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc13_22.2 [symbolic = %require_complete (constants.%require_complete.b7f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc13_22.2 [symbolic = %pattern_type (constants.%pattern_type.d48)]
// CHECK:STDOUT:   %array: @G.%array_type.loc13_22.2 (%array_type.281) = tuple_value () [symbolic = %array (constants.%array.2ed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @G.%pattern_type (%pattern_type.d48) = binding_pattern arr
// CHECK:STDOUT:       %.loc13_3.1: @G.%pattern_type (%pattern_type.d48) = var_pattern %arr.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc13_22.2 (%array_type.281) = var arr
// CHECK:STDOUT:     %.loc13_27.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_27.2: init @G.%array_type.loc13_22.2 (%array_type.281) = array_init () to %arr.var [symbolic = %array (constants.%array.2ed)]
// CHECK:STDOUT:     %.loc13_3.2: init @G.%array_type.loc13_22.2 (%array_type.281) = converted %.loc13_27.1, %.loc13_27.2 [symbolic = %array (constants.%array.2ed)]
// CHECK:STDOUT:     assign %arr.var, %.loc13_3.2
// CHECK:STDOUT:     %.loc13_22: type = splice_block %array_type.loc13_22.1 [symbolic = %array_type.loc13_22.2 (constants.%array_type.281)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc11_6.1 [symbolic = %T.loc11_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:       %array_type.loc13_22.1: type = array_type %int_0, %T.ref [symbolic = %array_type.loc13_22.2 (constants.%array_type.281)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc13_22.2 (%array_type.281) = bind_name arr, %arr.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %T.loc11_6.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc11_6.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %array_type.loc13_22.2 => constants.%array_type.f5c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cff
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.772
// CHECK:STDOUT:   %array => constants.%array.812
// CHECK:STDOUT: }
// CHECK:STDOUT:
