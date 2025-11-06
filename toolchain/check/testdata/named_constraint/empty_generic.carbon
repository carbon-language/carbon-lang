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

fn F(T:! type, U:! Empty(T)) {}

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
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Empty.type.a8a: type = generic_named_constaint_type @Empty [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_struct: %Empty.type.a8a = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.type.b8d23b.1: type = facet_type <@Empty, @Empty(%T.d9f)> [symbolic]
// CHECK:STDOUT:   %Self.aa1: %Empty.type.b8d23b.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U.aa1546.1: %Empty.type.b8d23b.1 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.e25e9f.1: type = pattern_type %Empty.type.b8d23b.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.e25e9f.2: type = pattern_type %type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.3b3: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.3b3 [symbolic]
// CHECK:STDOUT:   %Empty.type.b8d23b.2: type = facet_type <@Empty, @Empty(%T.binding.as_type)> [symbolic]
// CHECK:STDOUT:   %U.aa1546.2: %Empty.type.b8d23b.2 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.e25e9f.3: type = pattern_type %Empty.type.b8d23b.2 [symbolic]
// CHECK:STDOUT:   %V: type = symbolic_binding V, 2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %facet_value.6fa: %type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.8ba: <specific function> = specific_function %F, @F(%T.binding.as_type, %facet_value.6fa) [symbolic]
// CHECK:STDOUT:   %facet_value.832: %type = facet_value type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.55c: <specific function> = specific_function %F, @F(%T.binding.as_type, %facet_value.832) [symbolic]
// CHECK:STDOUT:   %facet_value.146: %type = facet_value %T.binding.as_type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.d3f: <specific function> = specific_function %F, @F(%T.binding.as_type, %facet_value.146) [symbolic]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.aa1546.2 [symbolic]
// CHECK:STDOUT:   %facet_value.356: %type = facet_value %U.binding.as_type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.df4: <specific function> = specific_function %F, @F(%T.binding.as_type, %facet_value.356) [symbolic]
// CHECK:STDOUT:   %facet_value.920: %type = facet_value %V, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.905: <specific function> = specific_function %F, @F(%T.binding.as_type, %facet_value.920) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Empty.decl: %Empty.type.a8a = constraint_decl @Empty [concrete = constants.%empty_struct] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc16_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc16_18.1 (constants.%T.d9f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @F.%pattern_type (%pattern_type.e25e9f.1) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc18_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc18_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:     %.loc18: type = splice_block %Empty.type [symbolic = %type (constants.%Empty.type.b8d23b.1)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: %Empty.type.a8a = name_ref Empty, file.%Empty.decl [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc18_6.2 [symbolic = %T.loc18_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:       %Empty.type: type = facet_type <@Empty, @Empty(constants.%T.d9f)> [symbolic = %type (constants.%Empty.type.b8d23b.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc18_16.2: @F.%type (%Empty.type.b8d23b.1) = symbolic_binding U, 1 [symbolic = %U.loc18_16.1 (constants.%U.aa1546.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.4a0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: @G.%pattern_type (%pattern_type.e25e9f.3) = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.98f = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_10: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20_6.2: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:     %.loc20_24.1: type = splice_block %Empty.type [symbolic = %type (constants.%Empty.type.b8d23b.2)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: %Empty.type.a8a = name_ref Empty, file.%Empty.decl [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %T.ref.loc20: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:       %T.as_type.loc20: type = facet_access_type %T.ref.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc20_24.2: type = converted %T.ref.loc20, %T.as_type.loc20 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %Empty.type: type = facet_type <@Empty, @Empty(constants.%T.binding.as_type)> [symbolic = %type (constants.%Empty.type.b8d23b.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc20_13.2: @G.%type (%Empty.type.b8d23b.2) = symbolic_binding U, 1 [symbolic = %U.loc20_13.1 (constants.%U.aa1546.2)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %V.loc20_27.2: type = symbolic_binding V, 2 [symbolic = %V.loc20_27.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @Empty(%T.loc16_18.2: type) {
// CHECK:STDOUT:   %T.loc16_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc16_18.1 (constants.%T.d9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %type: type = facet_type <type> [symbolic = %type (constants.%Empty.type.b8d23b.1)]
// CHECK:STDOUT:   %Self.loc16_28.2: @Empty.%type (%Empty.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self.loc16_28.2 (constants.%Self.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc16_28.1: @Empty.%type (%Empty.type.b8d23b.1) = symbolic_binding Self, 1 [symbolic = %Self.loc16_28.2 (constants.%Self.aa1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc16_28.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc18_6.2: type, %U.loc18_16.2: @F.%type (%Empty.type.b8d23b.1)) {
// CHECK:STDOUT:   %T.loc18_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc18_6.1 (constants.%T.d9f)]
// CHECK:STDOUT:   %type: type = facet_type <type> [symbolic = %type (constants.%Empty.type.b8d23b.1)]
// CHECK:STDOUT:   %U.loc18_16.1: @F.%type (%Empty.type.b8d23b.1) = symbolic_binding U, 1 [symbolic = %U.loc18_16.1 (constants.%U.aa1546.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %type [symbolic = %pattern_type (constants.%pattern_type.e25e9f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc20_6.2: %Z.type, %U.loc20_13.2: @G.%type (%Empty.type.b8d23b.2), %V.loc20_27.2: type) {
// CHECK:STDOUT:   %T.loc20_6.1: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc20_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %type: type = facet_type <type> [symbolic = %type (constants.%Empty.type.b8d23b.2)]
// CHECK:STDOUT:   %U.loc20_13.1: @G.%type (%Empty.type.b8d23b.2) = symbolic_binding U, 1 [symbolic = %U.loc20_13.1 (constants.%U.aa1546.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %type [symbolic = %pattern_type (constants.%pattern_type.e25e9f.3)]
// CHECK:STDOUT:   %V.loc20_27.1: type = symbolic_binding V, 2 [symbolic = %V.loc20_27.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.specific_fn.loc21_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, constants.%facet_value.6fa) [symbolic = %F.specific_fn.loc21_3.2 (constants.%F.specific_fn.8ba)]
// CHECK:STDOUT:   %F.specific_fn.loc22_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, constants.%facet_value.832) [symbolic = %F.specific_fn.loc22_3.2 (constants.%F.specific_fn.55c)]
// CHECK:STDOUT:   %facet_value.loc23_9.2: %type = facet_value %T.binding.as_type, () [symbolic = %facet_value.loc23_9.2 (constants.%facet_value.146)]
// CHECK:STDOUT:   %F.specific_fn.loc23_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %facet_value.loc23_9.2) [symbolic = %F.specific_fn.loc23_3.2 (constants.%F.specific_fn.d3f)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 1, %U.loc20_13.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:   %facet_value.loc24_9.2: %type = facet_value %U.binding.as_type, () [symbolic = %facet_value.loc24_9.2 (constants.%facet_value.356)]
// CHECK:STDOUT:   %F.specific_fn.loc24_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %facet_value.loc24_9.2) [symbolic = %F.specific_fn.loc24_3.2 (constants.%F.specific_fn.df4)]
// CHECK:STDOUT:   %facet_value.loc25_9.2: %type = facet_value %V.loc20_27.1, () [symbolic = %facet_value.loc25_9.2 (constants.%facet_value.920)]
// CHECK:STDOUT:   %F.specific_fn.loc25_3.2: <specific function> = specific_function constants.%F, @F(%T.binding.as_type, %facet_value.loc25_9.2) [symbolic = %F.specific_fn.loc25_3.2 (constants.%F.specific_fn.905)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref.loc21: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc21: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:     %.loc21_9: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc21_11: type = converted %.loc21_9, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %T.as_type.loc21: type = facet_access_type %T.ref.loc21 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc21_18.1: type = converted %T.ref.loc21, %T.as_type.loc21 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %facet_value.loc21: %type = facet_value constants.%empty_struct_type, () [concrete = constants.%facet_value.6fa]
// CHECK:STDOUT:     %.loc21_18.2: %type = converted constants.%empty_struct_type, %facet_value.loc21 [concrete = constants.%facet_value.6fa]
// CHECK:STDOUT:     %F.specific_fn.loc21_3.1: <specific function> = specific_function %F.ref.loc21, @F(constants.%T.binding.as_type, constants.%facet_value.6fa) [symbolic = %F.specific_fn.loc21_3.2 (constants.%F.specific_fn.8ba)]
// CHECK:STDOUT:     %F.call.loc21: init %empty_tuple.type = call %F.specific_fn.loc21_3.1()
// CHECK:STDOUT:     %F.ref.loc22: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc22: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:     %T.as_type.loc22: type = facet_access_type %T.ref.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc22_12.1: type = converted %T.ref.loc22, %T.as_type.loc22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %facet_value.loc22: %type = facet_value type, () [concrete = constants.%facet_value.832]
// CHECK:STDOUT:     %.loc22_12.2: %type = converted type, %facet_value.loc22 [concrete = constants.%facet_value.832]
// CHECK:STDOUT:     %F.specific_fn.loc22_3.1: <specific function> = specific_function %F.ref.loc22, @F(constants.%T.binding.as_type, constants.%facet_value.832) [symbolic = %F.specific_fn.loc22_3.2 (constants.%F.specific_fn.55c)]
// CHECK:STDOUT:     %F.call.loc22: init %empty_tuple.type = call %F.specific_fn.loc22_3.1()
// CHECK:STDOUT:     %F.ref.loc23: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc23_5: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:     %T.ref.loc23_8: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:     %T.as_type.loc23: type = facet_access_type %T.ref.loc23_5 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc23_9.1: type = converted %T.ref.loc23_5, %T.as_type.loc23 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %as_type.loc23: type = facet_access_type constants.%T.3b3 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %facet_value.loc23_9.1: %type = facet_value %as_type.loc23, () [symbolic = %facet_value.loc23_9.2 (constants.%facet_value.146)]
// CHECK:STDOUT:     %.loc23_9.2: %type = converted constants.%T.3b3, %facet_value.loc23_9.1 [symbolic = %facet_value.loc23_9.2 (constants.%facet_value.146)]
// CHECK:STDOUT:     %F.specific_fn.loc23_3.1: <specific function> = specific_function %F.ref.loc23, @F(constants.%T.binding.as_type, constants.%facet_value.146) [symbolic = %F.specific_fn.loc23_3.2 (constants.%F.specific_fn.d3f)]
// CHECK:STDOUT:     %F.call.loc23: init %empty_tuple.type = call %F.specific_fn.loc23_3.1()
// CHECK:STDOUT:     %F.ref.loc24: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc24: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:     %U.ref: @G.%type (%Empty.type.b8d23b.2) = name_ref U, %U.loc20_13.2 [symbolic = %U.loc20_13.1 (constants.%U.aa1546.2)]
// CHECK:STDOUT:     %T.as_type.loc24: type = facet_access_type %T.ref.loc24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc24_9.1: type = converted %T.ref.loc24, %T.as_type.loc24 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %as_type.loc24: type = facet_access_type constants.%U.aa1546.2 [symbolic = %U.binding.as_type (constants.%U.binding.as_type)]
// CHECK:STDOUT:     %facet_value.loc24_9.1: %type = facet_value %as_type.loc24, () [symbolic = %facet_value.loc24_9.2 (constants.%facet_value.356)]
// CHECK:STDOUT:     %.loc24_9.2: %type = converted constants.%U.aa1546.2, %facet_value.loc24_9.1 [symbolic = %facet_value.loc24_9.2 (constants.%facet_value.356)]
// CHECK:STDOUT:     %F.specific_fn.loc24_3.1: <specific function> = specific_function %F.ref.loc24, @F(constants.%T.binding.as_type, constants.%facet_value.356) [symbolic = %F.specific_fn.loc24_3.2 (constants.%F.specific_fn.df4)]
// CHECK:STDOUT:     %F.call.loc24: init %empty_tuple.type = call %F.specific_fn.loc24_3.1()
// CHECK:STDOUT:     %F.ref.loc25: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref.loc25: %Z.type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T.3b3)]
// CHECK:STDOUT:     %V.ref: type = name_ref V, %V.loc20_27.2 [symbolic = %V.loc20_27.1 (constants.%V)]
// CHECK:STDOUT:     %T.as_type.loc25: type = facet_access_type %T.ref.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc25_9.1: type = converted %T.ref.loc25, %T.as_type.loc25 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %facet_value.loc25_9.1: %type = facet_value constants.%V, () [symbolic = %facet_value.loc25_9.2 (constants.%facet_value.920)]
// CHECK:STDOUT:     %.loc25_9.2: %type = converted constants.%V, %facet_value.loc25_9.1 [symbolic = %facet_value.loc25_9.2 (constants.%facet_value.920)]
// CHECK:STDOUT:     %F.specific_fn.loc25_3.1: <specific function> = specific_function %F.ref.loc25, @F(constants.%T.binding.as_type, constants.%facet_value.920) [symbolic = %F.specific_fn.loc25_3.2 (constants.%F.specific_fn.905)]
// CHECK:STDOUT:     %F.call.loc25: init %empty_tuple.type = call %F.specific_fn.loc25_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.d9f) {
// CHECK:STDOUT:   %T.loc16_18.1 => constants.%T.d9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.d9f, constants.%U.aa1546.1) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.d9f
// CHECK:STDOUT:   %type => constants.%type
// CHECK:STDOUT:   %U.loc18_16.1 => constants.%U.aa1546.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e25e9f.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc16_18.1 => constants.%T.binding.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.3b3, constants.%U.aa1546.2, constants.%V) {
// CHECK:STDOUT:   %T.loc20_6.1 => constants.%T.3b3
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %type => constants.%type
// CHECK:STDOUT:   %U.loc20_13.1 => constants.%U.aa1546.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e25e9f.2
// CHECK:STDOUT:   %V.loc20_27.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%facet_value.6fa) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %type => constants.%type
// CHECK:STDOUT:   %U.loc18_16.1 => constants.%facet_value.6fa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e25e9f.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%facet_value.832) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %type => constants.%type
// CHECK:STDOUT:   %U.loc18_16.1 => constants.%facet_value.832
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e25e9f.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%facet_value.146) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %type => constants.%type
// CHECK:STDOUT:   %U.loc18_16.1 => constants.%facet_value.146
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e25e9f.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%facet_value.356) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %type => constants.%type
// CHECK:STDOUT:   %U.loc18_16.1 => constants.%facet_value.356
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e25e9f.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.binding.as_type, constants.%facet_value.920) {
// CHECK:STDOUT:   %T.loc18_6.1 => constants.%T.binding.as_type
// CHECK:STDOUT:   %type => constants.%type
// CHECK:STDOUT:   %U.loc18_16.1 => constants.%facet_value.920
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e25e9f.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
