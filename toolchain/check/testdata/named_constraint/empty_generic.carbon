// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/empty_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/empty_generic.carbon

interface Z {}

//@dump-sem-ir-begin
constraint Empty(T:! type) {}

fn F(T:! type, unused U:! Empty(T)) {}

fn G(T:! Z, U:! Empty(T), V:! type) {
  F(T, {} as type);
  F(T, type);
  F(T, T);
  F(T, U);
  F(T, V);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- empty_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Empty.type.19a: type = generic_named_constaint_type @Empty [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct.019: %Empty.type.19a = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.type.d682d6.1: type = facet_type <@Empty, @Empty(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.31c17c.1: %Empty.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.e1f642.1: %Empty.type.d682d6.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Empty.type.d682d6.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.ea3: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.22b: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.ea3 [symbolic]
// CHECK:STDOUT:   %Empty.type.d682d6.2: type = facet_type <@Empty, @Empty(%T.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %U.e1f642.2: %Empty.type.d682d6.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %Empty.type.d682d6.2 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct.a40: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.31c17c.2: %Empty.type.d682d6.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %empty_struct.type.facet: %type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %Empty.facet.76f: %Empty.type.d682d6.2 = facet_value %empty_struct_type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.5f9: <specific function> = specific_function %F, @F(%T.binding.as_type, %Empty.facet.76f) [symbolic]
// CHECK:STDOUT:   %facet_value.2cb: %type = facet_value type, () [concrete]
// CHECK:STDOUT:   %Empty.facet.2cb: %Empty.type.d682d6.2 = facet_value type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ff8: <specific function> = specific_function %F, @F(%T.binding.as_type, %Empty.facet.2cb) [symbolic]
// CHECK:STDOUT:   %Empty.facet.4f8: %Empty.type.d682d6.2 = facet_value %T.binding.as_type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.55f: <specific function> = specific_function %F, @F(%T.binding.as_type, %Empty.facet.4f8) [symbolic]
// CHECK:STDOUT:   %F.specific_fn.5c5: <specific function> = specific_function %F, @F(%T.binding.as_type, %U.e1f642.2) [symbolic]
// CHECK:STDOUT:   %facet_value.ed5: %type = facet_value %V, () [symbolic]
// CHECK:STDOUT:   %Empty.facet.ed5: %Empty.type.d682d6.2 = facet_value %V, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.bb1: <specific function> = specific_function %F, @F(%T.binding.as_type, %Empty.facet.ed5) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Empty.decl: %Empty.type.19a = constraint_decl @Empty [concrete = constants.%empty_struct.019] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_22.1: type = splice_block %.loc16_22.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc16_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc16_18.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @F.%pattern_type (%pattern_type.9a587c.1) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18_10.1: type = splice_block %.loc18_10.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc18_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc18_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc18_6.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc18_34: type = splice_block %Empty.type.loc18_34.2 [symbolic = %Empty.type.loc18_34.1 (constants.%Empty.type.d682d6.1)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: %Empty.type.19a = name_ref Empty, file.%Empty.decl [concrete = constants.%empty_struct.019]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc18_6.2 [symbolic = %T.loc18_6.1 (constants.%T.67d)]
// CHECK:STDOUT:       %Empty.type.loc18_34.2: type = facet_type <@Empty, @Empty(constants.%T.67d)> [symbolic = %Empty.type.loc18_34.1 (constants.%Empty.type.d682d6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc18_23.2: @F.%Empty.type.loc18_34.1 (%Empty.type.d682d6.1) = symbolic_binding U, 1 [symbolic = %U.loc18_23.1 (constants.%U.e1f642.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.22b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @G.%pattern_type (%pattern_type.9a587c.2) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_10: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20_6.2: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %.loc20_24.1: type = splice_block %Empty.type.loc20_24.2 [symbolic = %Empty.type.loc20_24.1 (constants.%Empty.type.d682d6.2)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: %Empty.type.19a = name_ref Empty, file.%Empty.decl [concrete = constants.%empty_struct.019]
// CHECK:STDOUT:       %T.ref.loc20: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:       %T.as_type.loc20: type = facet_access_type %T.ref.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc20_24.2: type = converted %T.ref.loc20, %T.as_type.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %Empty.type.loc20_24.2: type = facet_type <@Empty, @Empty(constants.%T.binding.as_type)> [symbolic = %Empty.type.loc20_24.1 (constants.%Empty.type.d682d6.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc20_13.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = symbolic_binding U, 1 [symbolic = %U.loc20_13.1 (constants.%U.e1f642.2)]
// CHECK:STDOUT:     %.loc20_31.1: type = splice_block %.loc20_31.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc20_31.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc20_27.2: type = symbolic_binding V, 2 [symbolic = %V.loc20_27.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @Empty(%T.loc16_18.2: type) {
// CHECK:STDOUT:   %T.loc16_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc16_18.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty, @Empty(%T.loc16_18.1)> [symbolic = %Empty.type (constants.%Empty.type.d682d6.1)]
// CHECK:STDOUT:   %Self.loc16_28.2: @Empty.%Empty.type (%Empty.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self.loc16_28.2 (constants.%Self.31c17c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc16_28.1: @Empty.%Empty.type (%Empty.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self.loc16_28.2 (constants.%Self.31c17c.1)]
// CHECK:STDOUT:     constraint_with_self_decl @Empty [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc16_28.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc18_6.2: type, %U.loc18_23.2: @F.%Empty.type.loc18_34.1 (%Empty.type.d682d6.1)) {
// CHECK:STDOUT:   %T.loc18_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc18_6.1 (constants.%T.67d)]
// CHECK:STDOUT:   %Empty.type.loc18_34.1: type = facet_type <@Empty, @Empty(%T.loc18_6.1)> [symbolic = %Empty.type.loc18_34.1 (constants.%Empty.type.d682d6.1)]
// CHECK:STDOUT:   %U.loc18_23.1: @F.%Empty.type.loc18_34.1 (%Empty.type.d682d6.1) = symbolic_binding U, 1 [symbolic = %U.loc18_23.1 (constants.%U.e1f642.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Empty.type.loc18_34.1 [symbolic = %pattern_type (constants.%pattern_type.9a587c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc20_6.2: %Z.type, %U.loc20_13.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2), %V.loc20_27.2: type) {
// CHECK:STDOUT:   %T.loc20_6.1: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc20_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %Empty.type.loc20_24.1: type = facet_type <@Empty, @Empty(%T.binding.as_type)> [symbolic = %Empty.type.loc20_24.1 (constants.%Empty.type.d682d6.2)]
// CHECK:STDOUT:   %U.loc20_13.1: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = symbolic_binding U, 1 [symbolic = %U.loc20_13.1 (constants.%U.e1f642.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Empty.type.loc20_24.1 [symbolic = %pattern_type (constants.%pattern_type.9a587c.2)]
// CHECK:STDOUT:   %V.loc20_27.1: type = symbolic_binding V, 2 [symbolic = %V.loc20_27.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.facet.loc21_18.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value constants.%empty_struct_type, () [symbolic = %Empty.facet.loc21_18.2 (constants.%Empty.facet.76f)]
// CHECK:STDOUT:   %F.specific_fn.loc21_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc21_18.2) [symbolic = %F.specific_fn.loc21_3.2 (constants.%F.specific_fn.5f9)]
// CHECK:STDOUT:   %Empty.facet.loc22_12.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value type, () [symbolic = %Empty.facet.loc22_12.2 (constants.%Empty.facet.2cb)]
// CHECK:STDOUT:   %F.specific_fn.loc22_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc22_12.2) [symbolic = %F.specific_fn.loc22_3.2 (constants.%F.specific_fn.ff8)]
// CHECK:STDOUT:   %Empty.facet.loc23_9.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value %T.binding.as_type, () [symbolic = %Empty.facet.loc23_9.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:   %F.specific_fn.loc23_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc23_9.2) [symbolic = %F.specific_fn.loc23_3.2 (constants.%F.specific_fn.55f)]
// CHECK:STDOUT:   %F.specific_fn.loc24_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %U.loc20_13.1) [symbolic = %F.specific_fn.loc24_3.2 (constants.%F.specific_fn.5c5)]
// CHECK:STDOUT:   %Empty.facet.loc25_9.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value %V.loc20_27.1, () [symbolic = %Empty.facet.loc25_9.2 (constants.%Empty.facet.ed5)]
// CHECK:STDOUT:   %F.specific_fn.loc25_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc25_9.2) [symbolic = %F.specific_fn.loc25_3.2 (constants.%F.specific_fn.bb1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref.loc21: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc21: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %.loc21_9: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:     %.loc21_14: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc21_11: type = converted %.loc21_9, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %T.as_type.loc21: type = facet_access_type %T.ref.loc21 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc21_18.1: type = converted %T.ref.loc21, %T.as_type.loc21 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc21_18.1: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value constants.%empty_struct_type, () [symbolic = %Empty.facet.loc21_18.2 (constants.%Empty.facet.76f)]
// CHECK:STDOUT:     %.loc21_18.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = converted constants.%empty_struct_type, %Empty.facet.loc21_18.1 [symbolic = %Empty.facet.loc21_18.2 (constants.%Empty.facet.76f)]
// CHECK:STDOUT:     %F.specific_fn.loc21_3.1: <specific function> = specific_function %F.ref.loc21, @F(constants.%T.binding.as_type, constants.%Empty.facet.76f) [symbolic = %F.specific_fn.loc21_3.2 (constants.%F.specific_fn.5f9)]
// CHECK:STDOUT:     %F.call.loc21: init %empty_tuple.type = call %F.specific_fn.loc21_3.1()
// CHECK:STDOUT:     %F.ref.loc22: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc22: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %.loc22_8: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc22_12.1: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc22_12.1: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value type, () [symbolic = %Empty.facet.loc22_12.2 (constants.%Empty.facet.2cb)]
// CHECK:STDOUT:     %.loc22_12.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = converted type, %Empty.facet.loc22_12.1 [symbolic = %Empty.facet.loc22_12.2 (constants.%Empty.facet.2cb)]
// CHECK:STDOUT:     %F.specific_fn.loc22_3.1: <specific function> = specific_function %F.ref.loc22, @F(constants.%T.binding.as_type, constants.%Empty.facet.2cb) [symbolic = %F.specific_fn.loc22_3.2 (constants.%F.specific_fn.ff8)]
// CHECK:STDOUT:     %F.call.loc22: init %empty_tuple.type = call %F.specific_fn.loc22_3.1()
// CHECK:STDOUT:     %F.ref.loc23: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc23_5: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %T.ref.loc23_8: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %T.as_type.loc23_9.1: type = facet_access_type %T.ref.loc23_5 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc23_9.1: type = converted %T.ref.loc23_5, %T.as_type.loc23_9.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %T.as_type.loc23_9.2: type = facet_access_type constants.%T.ea3 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc23_9.1: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value %T.as_type.loc23_9.2, () [symbolic = %Empty.facet.loc23_9.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:     %.loc23_9.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = converted constants.%T.ea3, %Empty.facet.loc23_9.1 [symbolic = %Empty.facet.loc23_9.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:     %F.specific_fn.loc23_3.1: <specific function> = specific_function %F.ref.loc23, @F(constants.%T.binding.as_type, constants.%Empty.facet.4f8) [symbolic = %F.specific_fn.loc23_3.2 (constants.%F.specific_fn.55f)]
// CHECK:STDOUT:     %F.call.loc23: init %empty_tuple.type = call %F.specific_fn.loc23_3.1()
// CHECK:STDOUT:     %F.ref.loc24: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc24: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %U.ref: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = name_ref U, %U.loc20_13.2 [symbolic = %U.loc20_13.1 (constants.%U.e1f642.2)]
// CHECK:STDOUT:     %T.as_type.loc24: type = facet_access_type %T.ref.loc24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc24: type = converted %T.ref.loc24, %T.as_type.loc24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %F.specific_fn.loc24_3.1: <specific function> = specific_function %F.ref.loc24, @F(constants.%T.binding.as_type, constants.%U.e1f642.2) [symbolic = %F.specific_fn.loc24_3.2 (constants.%F.specific_fn.5c5)]
// CHECK:STDOUT:     %F.call.loc24: init %empty_tuple.type = call %F.specific_fn.loc24_3.1()
// CHECK:STDOUT:     %F.ref.loc25: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc25: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %V.ref: type = name_ref V, %V.loc20_27.2 [symbolic = %V.loc20_27.1 (constants.%V)]
// CHECK:STDOUT:     %T.as_type.loc25: type = facet_access_type %T.ref.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc25_9.1: type = converted %T.ref.loc25, %T.as_type.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc25_9.1: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = facet_value constants.%V, () [symbolic = %Empty.facet.loc25_9.2 (constants.%Empty.facet.ed5)]
// CHECK:STDOUT:     %.loc25_9.2: @G.%Empty.type.loc20_24.1 (%Empty.type.d682d6.2) = converted constants.%V, %Empty.facet.loc25_9.1 [symbolic = %Empty.facet.loc25_9.2 (constants.%Empty.facet.ed5)]
// CHECK:STDOUT:     %F.specific_fn.loc25_3.1: <specific function> = specific_function %F.ref.loc25, @F(constants.%T.binding.as_type, constants.%Empty.facet.ed5) [symbolic = %F.specific_fn.loc25_3.2 (constants.%F.specific_fn.bb1)]
// CHECK:STDOUT:     %F.call.loc25: init %empty_tuple.type = call %F.specific_fn.loc25_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc16_18.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.67d, constants.%Self.31c17c.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d, constants.%U.e1f642.1) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.67d
// CHECK:STDOUT:   %Empty.type.loc18_34.1 => constants.%Empty.type.d682d6.1
// CHECK:STDOUT:   %U.loc18_23.1 => constants.%U.e1f642.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc16_18.1 => constants.%T.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %Self.loc16_28.2 => constants.%Self.31c17c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.ea3, constants.%U.e1f642.2, constants.%V) {
// CHECK:STDOUT:   %T.loc20_6.1 => constants.%T.ea3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc20_24.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc20_13.1 => constants.%U.e1f642.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:   %V.loc20_27.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type, constants.%Self.31c17c.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type, constants.%empty_struct.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.76f) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc18_34.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc18_23.1 => constants.%Empty.facet.76f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type, constants.%facet_value.2cb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.2cb) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc18_34.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc18_23.1 => constants.%Empty.facet.2cb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type, constants.%T.ea3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.4f8) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc18_34.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc18_23.1 => constants.%Empty.facet.4f8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%U.e1f642.2) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc18_34.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc18_23.1 => constants.%U.e1f642.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type, constants.%facet_value.ed5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.ed5) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc18_34.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc18_23.1 => constants.%Empty.facet.ed5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
