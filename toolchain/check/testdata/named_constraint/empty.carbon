// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/empty.carbon

constraint Empty {}

// No constraints, so it matches any type, but you can't do anything with it.
// Like `type`.
//
//@dump-sem-ir-begin
fn F(unused T:! Empty) {}
//@dump-sem-ir-end

interface Z {}

//@dump-sem-ir-begin
fn G(T:! Z, U:! type, V:! Empty) {
  F({});
  F(type);
  F(T);
  F(U);
  F(V);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %T.f45: %Empty.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %pattern_type.22b: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T.ea3: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %V: %Empty.type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.facet.76f: %Empty.type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.da7: <specific function> = specific_function %F, @F(%Empty.facet.76f) [concrete]
// CHECK:STDOUT:   %Empty.facet.2cb: %Empty.type = facet_value type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.9dd: <specific function> = specific_function %F, @F(%Empty.facet.2cb) [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.ea3 [symbolic]
// CHECK:STDOUT:   %Empty.facet.4f8: %Empty.type = facet_value %T.binding.as_type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.e9f: <specific function> = specific_function %F, @F(%Empty.facet.4f8) [symbolic]
// CHECK:STDOUT:   %Empty.facet.380: %Empty.type = facet_value %U, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ed0: <specific function> = specific_function %F, @F(%Empty.facet.380) [symbolic]
// CHECK:STDOUT:   %F.specific_fn.e30: <specific function> = specific_function %F, @F(%V) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11: type = splice_block %Empty.ref [concrete = constants.%Empty.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9.2: %Empty.type = symbolic_binding T, 0 [symbolic = %T.loc9.1 (constants.%T.f45)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.22b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.9a5 = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc22.2: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc22.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %.loc28.1: type = splice_block %.loc28.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc26.2: type = symbolic_binding U, 1 [symbolic = %U.loc26.1 (constants.%U)]
// CHECK:STDOUT:     %.loc32: type = splice_block %Empty.ref [concrete = constants.%Empty.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc30.2: %Empty.type = symbolic_binding V, 2 [symbolic = %V.loc30.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9.2: %Empty.type) {
// CHECK:STDOUT:   %T.loc9.1: %Empty.type = symbolic_binding T, 0 [symbolic = %T.loc9.1 (constants.%T.f45)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc22.2: %Z.type, %U.loc26.2: type, %V.loc30.2: %Empty.type) {
// CHECK:STDOUT:   %T.loc22.1: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc22.1 (constants.%T.ea3)]
// CHECK:STDOUT:   %U.loc26.1: type = symbolic_binding U, 1 [symbolic = %U.loc26.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc30.1: %Empty.type = symbolic_binding V, 2 [symbolic = %V.loc30.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc22.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %Empty.facet.loc49.2: %Empty.type = facet_value %T.binding.as_type, () [symbolic = %Empty.facet.loc49.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:   %F.specific_fn.loc46.2: <specific function> = specific_function constants.%F, @F(%Empty.facet.loc49.2) [symbolic = %F.specific_fn.loc46.2 (constants.%F.specific_fn.e9f)]
// CHECK:STDOUT:   %Empty.facet.loc54.2: %Empty.type = facet_value %U.loc26.1, () [symbolic = %Empty.facet.loc54.2 (constants.%Empty.facet.380)]
// CHECK:STDOUT:   %F.specific_fn.loc51.2: <specific function> = specific_function constants.%F, @F(%Empty.facet.loc54.2) [symbolic = %F.specific_fn.loc51.2 (constants.%F.specific_fn.ed0)]
// CHECK:STDOUT:   %F.specific_fn.loc56.2: <specific function> = specific_function constants.%F, @F(%V.loc30.1) [symbolic = %F.specific_fn.loc56.2 (constants.%F.specific_fn.e30)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref.loc35: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc38: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %Empty.facet.loc39: %Empty.type = facet_value constants.%empty_struct_type, () [concrete = constants.%Empty.facet.76f]
// CHECK:STDOUT:     %.loc39: %Empty.type = converted %.loc38, %Empty.facet.loc39 [concrete = constants.%Empty.facet.76f]
// CHECK:STDOUT:     %F.specific_fn.loc35: <specific function> = specific_function %F.ref.loc35, @F(constants.%Empty.facet.76f) [concrete = constants.%F.specific_fn.da7]
// CHECK:STDOUT:     %F.call.loc39: init %empty_tuple.type = call %F.specific_fn.loc35()
// CHECK:STDOUT:     %F.ref.loc41: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc43: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %Empty.facet.loc44: %Empty.type = facet_value %.loc43, () [concrete = constants.%Empty.facet.2cb]
// CHECK:STDOUT:     %.loc44: %Empty.type = converted %.loc43, %Empty.facet.loc44 [concrete = constants.%Empty.facet.2cb]
// CHECK:STDOUT:     %F.specific_fn.loc41: <specific function> = specific_function %F.ref.loc41, @F(constants.%Empty.facet.2cb) [concrete = constants.%F.specific_fn.9dd]
// CHECK:STDOUT:     %F.call.loc44: init %empty_tuple.type = call %F.specific_fn.loc41()
// CHECK:STDOUT:     %F.ref.loc46: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %Z.type = name_ref T, %T.loc22.2 [symbolic = %T.loc22.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc49.1: %Empty.type = facet_value %T.as_type, () [symbolic = %Empty.facet.loc49.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:     %.loc49: %Empty.type = converted %T.ref, %Empty.facet.loc49.1 [symbolic = %Empty.facet.loc49.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:     %F.specific_fn.loc46.1: <specific function> = specific_function %F.ref.loc46, @F(constants.%Empty.facet.4f8) [symbolic = %F.specific_fn.loc46.2 (constants.%F.specific_fn.e9f)]
// CHECK:STDOUT:     %F.call.loc49: init %empty_tuple.type = call %F.specific_fn.loc46.1()
// CHECK:STDOUT:     %F.ref.loc51: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc26.2 [symbolic = %U.loc26.1 (constants.%U)]
// CHECK:STDOUT:     %Empty.facet.loc54.1: %Empty.type = facet_value %U.ref, () [symbolic = %Empty.facet.loc54.2 (constants.%Empty.facet.380)]
// CHECK:STDOUT:     %.loc54: %Empty.type = converted %U.ref, %Empty.facet.loc54.1 [symbolic = %Empty.facet.loc54.2 (constants.%Empty.facet.380)]
// CHECK:STDOUT:     %F.specific_fn.loc51.1: <specific function> = specific_function %F.ref.loc51, @F(constants.%Empty.facet.380) [symbolic = %F.specific_fn.loc51.2 (constants.%F.specific_fn.ed0)]
// CHECK:STDOUT:     %F.call.loc54: init %empty_tuple.type = call %F.specific_fn.loc51.1()
// CHECK:STDOUT:     %F.ref.loc56: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %V.ref: %Empty.type = name_ref V, %V.loc30.2 [symbolic = %V.loc30.1 (constants.%V)]
// CHECK:STDOUT:     %F.specific_fn.loc56.1: <specific function> = specific_function %F.ref.loc56, @F(constants.%V) [symbolic = %F.specific_fn.loc56.2 (constants.%F.specific_fn.e30)]
// CHECK:STDOUT:     %F.call.loc59: init %empty_tuple.type = call %F.specific_fn.loc56.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.f45) {
// CHECK:STDOUT:   %T.loc9.1 => constants.%T.f45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.ea3, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.loc22.1 => constants.%T.ea3
// CHECK:STDOUT:   %U.loc26.1 => constants.%U
// CHECK:STDOUT:   %V.loc30.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Empty.facet.76f) {
// CHECK:STDOUT:   %T.loc9.1 => constants.%Empty.facet.76f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Empty.facet.2cb) {
// CHECK:STDOUT:   %T.loc9.1 => constants.%Empty.facet.2cb
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Empty.facet.4f8) {
// CHECK:STDOUT:   %T.loc9.1 => constants.%Empty.facet.4f8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Empty.facet.380) {
// CHECK:STDOUT:   %T.loc9.1 => constants.%Empty.facet.380
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%V) {
// CHECK:STDOUT:   %T.loc9.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
