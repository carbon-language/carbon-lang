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
fn F(T:! Empty) {}
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %T.85e: %type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e25: type = pattern_type %type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %T.1d7: %Z.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %V: %type = bind_symbolic_name V, 2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %facet_value.6fa: %type = facet_value %empty_struct_type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.007: <specific function> = specific_function %F, @F(%facet_value.6fa) [concrete]
// CHECK:STDOUT:   %facet_value.832: %type = facet_value type, () [concrete]
// CHECK:STDOUT:   %F.specific_fn.7db: <specific function> = specific_function %F, @F(%facet_value.832) [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.1d7 [symbolic]
// CHECK:STDOUT:   %facet_value.f04: %type = facet_value %T.binding.as_type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.59e: <specific function> = specific_function %F, @F(%facet_value.f04) [symbolic]
// CHECK:STDOUT:   %facet_value.137: %type = facet_value %U, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.296: <specific function> = specific_function %F, @F(%facet_value.137) [symbolic]
// CHECK:STDOUT:   %F.specific_fn.88e: <specific function> = specific_function %F, @F(%V) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e25 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19: type = splice_block %Empty.ref [concrete = constants.%type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19_6.2: %type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.1 (constants.%T.85e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.4a0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %V.patt: %pattern_type.e25 = symbolic_binding_pattern V, 2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25_10: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc25_6.2: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc25_6.1 (constants.%T.1d7)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %U.loc25_13.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc25_13.1 (constants.%U)]
// CHECK:STDOUT:     %.loc25_27: type = splice_block %Empty.ref [concrete = constants.%type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %V.loc25_23.2: %type = bind_symbolic_name V, 2 [symbolic = %V.loc25_23.1 (constants.%V)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc19_6.2: %type) {
// CHECK:STDOUT:   %T.loc19_6.1: %type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.1 (constants.%T.85e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc25_6.2: %Z.type, %U.loc25_13.2: type, %V.loc25_23.2: %type) {
// CHECK:STDOUT:   %T.loc25_6.1: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc25_6.1 (constants.%T.1d7)]
// CHECK:STDOUT:   %U.loc25_13.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc25_13.1 (constants.%U)]
// CHECK:STDOUT:   %V.loc25_23.1: %type = bind_symbolic_name V, 2 [symbolic = %V.loc25_23.1 (constants.%V)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc25_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %facet_value.loc28_6.2: %type = facet_value %T.binding.as_type, () [symbolic = %facet_value.loc28_6.2 (constants.%facet_value.f04)]
// CHECK:STDOUT:   %F.specific_fn.loc28_3.2: <specific function> = specific_function constants.%F, @F(%facet_value.loc28_6.2) [symbolic = %F.specific_fn.loc28_3.2 (constants.%F.specific_fn.59e)]
// CHECK:STDOUT:   %facet_value.loc29_6.2: %type = facet_value %U.loc25_13.1, () [symbolic = %facet_value.loc29_6.2 (constants.%facet_value.137)]
// CHECK:STDOUT:   %F.specific_fn.loc29_3.2: <specific function> = specific_function constants.%F, @F(%facet_value.loc29_6.2) [symbolic = %F.specific_fn.loc29_3.2 (constants.%F.specific_fn.296)]
// CHECK:STDOUT:   %F.specific_fn.loc30_3.2: <specific function> = specific_function constants.%F, @F(%V.loc25_23.1) [symbolic = %F.specific_fn.loc30_3.2 (constants.%F.specific_fn.88e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref.loc26: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %.loc26_6: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %facet_value.loc26: %type = facet_value constants.%empty_struct_type, () [concrete = constants.%facet_value.6fa]
// CHECK:STDOUT:     %.loc26_7: %type = converted %.loc26_6, %facet_value.loc26 [concrete = constants.%facet_value.6fa]
// CHECK:STDOUT:     %F.specific_fn.loc26: <specific function> = specific_function %F.ref.loc26, @F(constants.%facet_value.6fa) [concrete = constants.%F.specific_fn.007]
// CHECK:STDOUT:     %F.call.loc26: init %empty_tuple.type = call %F.specific_fn.loc26()
// CHECK:STDOUT:     %F.ref.loc27: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %facet_value.loc27: %type = facet_value type, () [concrete = constants.%facet_value.832]
// CHECK:STDOUT:     %.loc27: %type = converted type, %facet_value.loc27 [concrete = constants.%facet_value.832]
// CHECK:STDOUT:     %F.specific_fn.loc27: <specific function> = specific_function %F.ref.loc27, @F(constants.%facet_value.832) [concrete = constants.%F.specific_fn.7db]
// CHECK:STDOUT:     %F.call.loc27: init %empty_tuple.type = call %F.specific_fn.loc27()
// CHECK:STDOUT:     %F.ref.loc28: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %Z.type = name_ref T, %T.loc25_6.2 [symbolic = %T.loc25_6.1 (constants.%T.1d7)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %facet_value.loc28_6.1: %type = facet_value %T.as_type, () [symbolic = %facet_value.loc28_6.2 (constants.%facet_value.f04)]
// CHECK:STDOUT:     %.loc28: %type = converted %T.ref, %facet_value.loc28_6.1 [symbolic = %facet_value.loc28_6.2 (constants.%facet_value.f04)]
// CHECK:STDOUT:     %F.specific_fn.loc28_3.1: <specific function> = specific_function %F.ref.loc28, @F(constants.%facet_value.f04) [symbolic = %F.specific_fn.loc28_3.2 (constants.%F.specific_fn.59e)]
// CHECK:STDOUT:     %F.call.loc28: init %empty_tuple.type = call %F.specific_fn.loc28_3.1()
// CHECK:STDOUT:     %F.ref.loc29: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc25_13.2 [symbolic = %U.loc25_13.1 (constants.%U)]
// CHECK:STDOUT:     %facet_value.loc29_6.1: %type = facet_value %U.ref, () [symbolic = %facet_value.loc29_6.2 (constants.%facet_value.137)]
// CHECK:STDOUT:     %.loc29: %type = converted %U.ref, %facet_value.loc29_6.1 [symbolic = %facet_value.loc29_6.2 (constants.%facet_value.137)]
// CHECK:STDOUT:     %F.specific_fn.loc29_3.1: <specific function> = specific_function %F.ref.loc29, @F(constants.%facet_value.137) [symbolic = %F.specific_fn.loc29_3.2 (constants.%F.specific_fn.296)]
// CHECK:STDOUT:     %F.call.loc29: init %empty_tuple.type = call %F.specific_fn.loc29_3.1()
// CHECK:STDOUT:     %F.ref.loc30: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %V.ref: %type = name_ref V, %V.loc25_23.2 [symbolic = %V.loc25_23.1 (constants.%V)]
// CHECK:STDOUT:     %F.specific_fn.loc30_3.1: <specific function> = specific_function %F.ref.loc30, @F(constants.%V) [symbolic = %F.specific_fn.loc30_3.2 (constants.%F.specific_fn.88e)]
// CHECK:STDOUT:     %F.call.loc30: init %empty_tuple.type = call %F.specific_fn.loc30_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.85e) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%T.85e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.1d7, constants.%U, constants.%V) {
// CHECK:STDOUT:   %T.loc25_6.1 => constants.%T.1d7
// CHECK:STDOUT:   %U.loc25_13.1 => constants.%U
// CHECK:STDOUT:   %V.loc25_23.1 => constants.%V
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value.6fa) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%facet_value.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value.832) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%facet_value.832
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value.f04) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%facet_value.f04
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%facet_value.137) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%facet_value.137
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%V) {
// CHECK:STDOUT:   %T.loc19_6.1 => constants.%V
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
