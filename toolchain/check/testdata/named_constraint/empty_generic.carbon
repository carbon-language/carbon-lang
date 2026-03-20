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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Empty.type.19a: type = generic_named_constaint_type @Empty [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct.019: %Empty.type.19a = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.type.d682d6.1: type = facet_type <@Empty, @Empty(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.31c17c.1: %Empty.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type %Empty.type.d682d6.1 [symbolic]
// CHECK:STDOUT:   %U.e1f642.1: %Empty.type.d682d6.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.22b: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T.ea3: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.ea3 [symbolic]
// CHECK:STDOUT:   %Empty.type.d682d6.2: type = facet_type <@Empty, @Empty(%T.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %Empty.type.d682d6.2 [symbolic]
// CHECK:STDOUT:   %U.e1f642.2: %Empty.type.d682d6.2 = symbolic_binding U, 1 [symbolic]
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
// CHECK:STDOUT:     %.loc10.1: type = splice_block %.loc10.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8.2: type = symbolic_binding T, 0 [symbolic = %T.loc8.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @F.%pattern_type (%pattern_type.9a587c.1) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19.1: type = splice_block %.loc19.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17.2: type = symbolic_binding T, 0 [symbolic = %T.loc17.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc27: type = splice_block %Empty.type.loc27.2 [symbolic = %Empty.type.loc27.1 (constants.%Empty.type.d682d6.1)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: %Empty.type.19a = name_ref Empty, file.%Empty.decl [concrete = constants.%empty_struct.019]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc17.2 [symbolic = %T.loc17.1 (constants.%T.67d)]
// CHECK:STDOUT:       %Empty.type.loc27.2: type = facet_type <@Empty, @Empty(constants.%T.67d)> [symbolic = %Empty.type.loc27.1 (constants.%Empty.type.d682d6.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc22.2: @F.%Empty.type.loc27.1 (%Empty.type.d682d6.1) = symbolic_binding U, 1 [symbolic = %U.loc22.1 (constants.%U.e1f642.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.22b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @G.%pattern_type (%pattern_type.9a587c.2) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc36: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc34.2: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %.loc43.1: type = splice_block %Empty.type.loc43.2 [symbolic = %Empty.type.loc43.1 (constants.%Empty.type.d682d6.2)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: %Empty.type.19a = name_ref Empty, file.%Empty.decl [concrete = constants.%empty_struct.019]
// CHECK:STDOUT:       %T.ref.loc42: %Z.type = name_ref T, %T.loc34.2 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:       %T.as_type.loc43: type = facet_access_type %T.ref.loc42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc43.2: type = converted %T.ref.loc42, %T.as_type.loc43 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %Empty.type.loc43.2: type = facet_type <@Empty, @Empty(constants.%T.binding.as_type)> [symbolic = %Empty.type.loc43.1 (constants.%Empty.type.d682d6.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc38.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = symbolic_binding U, 1 [symbolic = %U.loc38.1 (constants.%U.e1f642.2)]
// CHECK:STDOUT:     %.loc47.1: type = splice_block %.loc47.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc47.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc45.2: type = symbolic_binding V, 2 [symbolic = %V.loc45.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @Empty(%T.loc8.2: type) {
// CHECK:STDOUT:   %T.loc8.1: type = symbolic_binding T, 0 [symbolic = %T.loc8.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty, @Empty(%T.loc8.1)> [symbolic = %Empty.type (constants.%Empty.type.d682d6.1)]
// CHECK:STDOUT:   %Self.loc12.2: @Empty.%Empty.type (%Empty.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self.loc12.2 (constants.%Self.31c17c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc12.1: @Empty.%Empty.type (%Empty.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self.loc12.2 (constants.%Self.31c17c.1)]
// CHECK:STDOUT:     %Empty.WithSelf.decl = constraint_with_self_decl @Empty [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc12.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc17.2: type, %U.loc22.2: @F.%Empty.type.loc27.1 (%Empty.type.d682d6.1)) {
// CHECK:STDOUT:   %T.loc17.1: type = symbolic_binding T, 0 [symbolic = %T.loc17.1 (constants.%T.67d)]
// CHECK:STDOUT:   %Empty.type.loc27.1: type = facet_type <@Empty, @Empty(%T.loc17.1)> [symbolic = %Empty.type.loc27.1 (constants.%Empty.type.d682d6.1)]
// CHECK:STDOUT:   %U.loc22.1: @F.%Empty.type.loc27.1 (%Empty.type.d682d6.1) = symbolic_binding U, 1 [symbolic = %U.loc22.1 (constants.%U.e1f642.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Empty.type.loc27.1 [symbolic = %pattern_type (constants.%pattern_type.9a587c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc34.2: %Z.type, %U.loc38.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2), %V.loc45.2: type) {
// CHECK:STDOUT:   %T.loc34.1: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc34.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %Empty.type.loc43.1: type = facet_type <@Empty, @Empty(%T.binding.as_type)> [symbolic = %Empty.type.loc43.1 (constants.%Empty.type.d682d6.2)]
// CHECK:STDOUT:   %U.loc38.1: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = symbolic_binding U, 1 [symbolic = %U.loc38.1 (constants.%U.e1f642.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Empty.type.loc43.1 [symbolic = %pattern_type (constants.%pattern_type.9a587c.2)]
// CHECK:STDOUT:   %V.loc45.1: type = symbolic_binding V, 2 [symbolic = %V.loc45.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.facet.loc58.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value constants.%empty_struct_type, () [symbolic = %Empty.facet.loc58.2 (constants.%Empty.facet.76f)]
// CHECK:STDOUT:   %F.specific_fn.loc50.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc58.2) [symbolic = %F.specific_fn.loc50.2 (constants.%F.specific_fn.5f9)]
// CHECK:STDOUT:   %Empty.facet.loc65.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value type, () [symbolic = %Empty.facet.loc65.2 (constants.%Empty.facet.2cb)]
// CHECK:STDOUT:   %F.specific_fn.loc60.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc65.2) [symbolic = %F.specific_fn.loc60.2 (constants.%F.specific_fn.ff8)]
// CHECK:STDOUT:   %Empty.facet.loc72.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value %T.binding.as_type, () [symbolic = %Empty.facet.loc72.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:   %F.specific_fn.loc67.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc72.2) [symbolic = %F.specific_fn.loc67.2 (constants.%F.specific_fn.55f)]
// CHECK:STDOUT:   %F.specific_fn.loc74.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %U.loc38.1) [symbolic = %F.specific_fn.loc74.2 (constants.%F.specific_fn.5c5)]
// CHECK:STDOUT:   %Empty.facet.loc86.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value %V.loc45.1, () [symbolic = %Empty.facet.loc86.2 (constants.%Empty.facet.ed5)]
// CHECK:STDOUT:   %F.specific_fn.loc81.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %Empty.facet.loc86.2) [symbolic = %F.specific_fn.loc81.2 (constants.%F.specific_fn.bb1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref.loc50: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc52: %Z.type = name_ref T, %T.loc34.2 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %.loc55: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct.a40]
// CHECK:STDOUT:     %.loc57: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc56: type = converted %.loc55, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %T.as_type.loc58: type = facet_access_type %T.ref.loc52 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc58.1: type = converted %T.ref.loc52, %T.as_type.loc58 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc58.1: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value constants.%empty_struct_type, () [symbolic = %Empty.facet.loc58.2 (constants.%Empty.facet.76f)]
// CHECK:STDOUT:     %.loc58.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = converted constants.%empty_struct_type, %Empty.facet.loc58.1 [symbolic = %Empty.facet.loc58.2 (constants.%Empty.facet.76f)]
// CHECK:STDOUT:     %F.specific_fn.loc50.1: <specific function> = specific_function %F.ref.loc50, @F(constants.%T.binding.as_type, constants.%Empty.facet.76f) [symbolic = %F.specific_fn.loc50.2 (constants.%F.specific_fn.5f9)]
// CHECK:STDOUT:     %F.call.loc58: init %empty_tuple.type = call %F.specific_fn.loc50.1()
// CHECK:STDOUT:     %F.ref.loc60: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc62: %Z.type = name_ref T, %T.loc34.2 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %.loc64: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %T.as_type.loc65: type = facet_access_type %T.ref.loc62 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc65.1: type = converted %T.ref.loc62, %T.as_type.loc65 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc65.1: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value type, () [symbolic = %Empty.facet.loc65.2 (constants.%Empty.facet.2cb)]
// CHECK:STDOUT:     %.loc65.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = converted type, %Empty.facet.loc65.1 [symbolic = %Empty.facet.loc65.2 (constants.%Empty.facet.2cb)]
// CHECK:STDOUT:     %F.specific_fn.loc60.1: <specific function> = specific_function %F.ref.loc60, @F(constants.%T.binding.as_type, constants.%Empty.facet.2cb) [symbolic = %F.specific_fn.loc60.2 (constants.%F.specific_fn.ff8)]
// CHECK:STDOUT:     %F.call.loc65: init %empty_tuple.type = call %F.specific_fn.loc60.1()
// CHECK:STDOUT:     %F.ref.loc67: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc69: %Z.type = name_ref T, %T.loc34.2 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %T.ref.loc71: %Z.type = name_ref T, %T.loc34.2 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %T.as_type.loc72.1: type = facet_access_type %T.ref.loc69 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc72.1: type = converted %T.ref.loc69, %T.as_type.loc72.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %T.as_type.loc72.2: type = facet_access_type constants.%T.ea3 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc72.1: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value %T.as_type.loc72.2, () [symbolic = %Empty.facet.loc72.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:     %.loc72.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = converted constants.%T.ea3, %Empty.facet.loc72.1 [symbolic = %Empty.facet.loc72.2 (constants.%Empty.facet.4f8)]
// CHECK:STDOUT:     %F.specific_fn.loc67.1: <specific function> = specific_function %F.ref.loc67, @F(constants.%T.binding.as_type, constants.%Empty.facet.4f8) [symbolic = %F.specific_fn.loc67.2 (constants.%F.specific_fn.55f)]
// CHECK:STDOUT:     %F.call.loc72: init %empty_tuple.type = call %F.specific_fn.loc67.1()
// CHECK:STDOUT:     %F.ref.loc74: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc76: %Z.type = name_ref T, %T.loc34.2 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %U.ref: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = name_ref U, %U.loc38.2 [symbolic = %U.loc38.1 (constants.%U.e1f642.2)]
// CHECK:STDOUT:     %T.as_type.loc79: type = facet_access_type %T.ref.loc76 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc79: type = converted %T.ref.loc76, %T.as_type.loc79 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %F.specific_fn.loc74.1: <specific function> = specific_function %F.ref.loc74, @F(constants.%T.binding.as_type, constants.%U.e1f642.2) [symbolic = %F.specific_fn.loc74.2 (constants.%F.specific_fn.5c5)]
// CHECK:STDOUT:     %F.call.loc79: init %empty_tuple.type = call %F.specific_fn.loc74.1()
// CHECK:STDOUT:     %F.ref.loc81: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc83: %Z.type = name_ref T, %T.loc34.2 [symbolic = %T.loc34.1 (constants.%T.ea3)]
// CHECK:STDOUT:     %V.ref: type = name_ref V, %V.loc45.2 [symbolic = %V.loc45.1 (constants.%V)]
// CHECK:STDOUT:     %T.as_type.loc86: type = facet_access_type %T.ref.loc83 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc86.1: type = converted %T.ref.loc83, %T.as_type.loc86 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Empty.facet.loc86.1: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = facet_value constants.%V, () [symbolic = %Empty.facet.loc86.2 (constants.%Empty.facet.ed5)]
// CHECK:STDOUT:     %.loc86.2: @G.%Empty.type.loc43.1 (%Empty.type.d682d6.2) = converted constants.%V, %Empty.facet.loc86.1 [symbolic = %Empty.facet.loc86.2 (constants.%Empty.facet.ed5)]
// CHECK:STDOUT:     %F.specific_fn.loc81.1: <specific function> = specific_function %F.ref.loc81, @F(constants.%T.binding.as_type, constants.%Empty.facet.ed5) [symbolic = %F.specific_fn.loc81.2 (constants.%F.specific_fn.bb1)]
// CHECK:STDOUT:     %F.call.loc86: init %empty_tuple.type = call %F.specific_fn.loc81.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc8.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T.67d, constants.%Self.31c17c.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d, constants.%U.e1f642.1) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%T.67d
// CHECK:STDOUT:   %Empty.type.loc27.1 => constants.%Empty.type.d682d6.1
// CHECK:STDOUT:   %U.loc22.1 => constants.%U.e1f642.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc8.1 => constants.%T.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Empty.type => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %Self.loc12.2 => constants.%Self.31c17c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.ea3, constants.%U.e1f642.2, constants.%V) {
// CHECK:STDOUT:   %T.loc34.1 => constants.%T.ea3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc43.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc38.1 => constants.%U.e1f642.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:   %V.loc45.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T.binding.as_type, constants.%Self.31c17c.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T.binding.as_type, constants.%empty_struct.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.76f) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc27.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc22.1 => constants.%Empty.facet.76f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T.binding.as_type, constants.%facet_value.2cb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.2cb) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc27.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc22.1 => constants.%Empty.facet.2cb
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T.binding.as_type, constants.%T.ea3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.4f8) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc27.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc22.1 => constants.%Empty.facet.4f8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%U.e1f642.2) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc27.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc22.1 => constants.%U.e1f642.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%T.binding.as_type, constants.%facet_value.ed5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%Empty.facet.ed5) {
// CHECK:STDOUT:   %T.loc17.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %Empty.type.loc27.1 => constants.%Empty.type.d682d6.2
// CHECK:STDOUT:   %U.loc22.1 => constants.%Empty.facet.ed5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a587c.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
