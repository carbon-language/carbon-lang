// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/import_specific.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/import_specific.carbon

// --- library.carbon

library "[[@TEST_NAME]]";

fn F(T:! type) {
}

fn G(T:! type) {
  F(T);
}

// --- user.carbon

library "[[@TEST_NAME]]";

import library "library";

class C {}

fn H() {
  // TODO: We currently crash when importing G if we don't force F to be imported first.
  F(C);

  G(C);
}

// CHECK:STDOUT: --- library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc7_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.1: type) {
// CHECK:STDOUT:   %T.loc4_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc4_6.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.specific_fn.loc8_3.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.2) [symbolic = %F.specific_fn.loc8_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.loc7_6.1: type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %F.specific_fn.loc8_3.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc8_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc8_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_6.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@G.%T.loc7_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//library, F, loaded [concrete = constants.%F]
// CHECK:STDOUT:   %Main.G: %G.type = import_ref Main//library, G, loaded [concrete = constants.%G]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//library, loc4_6, loaded [symbolic = @F.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//library, loc7_6, loaded [symbolic = @G.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .G = imports.%Main.G
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %C.ref.loc10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, imports.%Main.G [concrete = constants.%G]
// CHECK:STDOUT:   %C.ref.loc12: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%C) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(imports.%Main.import_ref.5ab3ec.1: type) [from "library.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.1: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(imports.%Main.import_ref.5ab3ec.2: type) [from "library.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function constants.%F, @F(%T) [symbolic = %F.specific_fn (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.patt.1: type);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@G.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %T.patt.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.specific_fn => constants.%F.specific_fn.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
