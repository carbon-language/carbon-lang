// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/convert.carbon

// --- type_and_with_named_constraint_self.carbon
library "[[@TEST_NAME]]";

constraint E {}

fn F(unused T:! E) {}

//@dump-sem-ir-begin
fn G(T:! E & E) {
  F(T);
}
//@dump-sem-ir-end

// --- compatible_constraints.carbon
library "[[@TEST_NAME]]";

constraint E1 {}
constraint E2 {}

fn F(unused T:! E1) {}

//@dump-sem-ir-begin
fn G(T:! E2) {
  F(T);
}
//@dump-sem-ir-end

// --- unspecified.carbon
library "[[@TEST_NAME]]";

interface Z {}

constraint E {
  require impls Z;
}

fn F(unused T:! E) {}

class C {}
impl C as Z {}

fn G(T:! Z) {
  // Anything that impls Z also impls E, since that's all it requires. Nothing
  // needs to specify that T impls E.
  F(T);
  // Same for concrete types.
  F(C);
}

// CHECK:STDOUT: --- type_and_with_named_constraint_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E.type: type = facet_type <@E> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type %E.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.4bd: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d15: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.4bd, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %E.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %T.f45530.2: %E.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T.f45530.2) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.8d3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.8d3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %.loc8_12.3 [concrete = constants.%E.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %E.ref.loc8_10: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %E.ref.loc8_14: type = name_ref E, file.%E.decl [concrete = constants.%E.type]
// CHECK:STDOUT:       %impl.elem0: %.d15 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %E.ref.loc8_10, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%E.ref.loc8_10, %E.ref.loc8_14) [concrete = constants.%E.type]
// CHECK:STDOUT:       %.loc8_12.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%E.type]
// CHECK:STDOUT:       %.loc8_12.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc8_12.2 [concrete = constants.%E.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_7.2: %E.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.f45530.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc8_7.2: %E.type) {
// CHECK:STDOUT:   %T.loc8_7.1: %E.type = symbolic_binding T, 0 [symbolic = %T.loc8_7.1 (constants.%T.f45530.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.specific_fn.loc9_3.2: <specific function> = specific_function constants.%F, @F(%T.loc8_7.1) [symbolic = %F.specific_fn.loc9_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %E.type = name_ref T, %T.loc8_7.2 [symbolic = %T.loc8_7.1 (constants.%T.f45530.2)]
// CHECK:STDOUT:     %F.specific_fn.loc9_3.1: <specific function> = specific_function %F.ref, @F(constants.%T.f45530.2) [symbolic = %F.specific_fn.loc9_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc9_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.f45530.2) {
// CHECK:STDOUT:   %T.loc8_7.1 => constants.%T.f45530.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compatible_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %E1.type: type = facet_type <@E1> [concrete]
// CHECK:STDOUT:   %E2.type: type = facet_type <@E2> [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9a587c.2: type = pattern_type %E2.type [concrete]
// CHECK:STDOUT:   %T.f45530.2: %E2.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f45530.2 [symbolic]
// CHECK:STDOUT:   %E1.facet: %E1.type = facet_value %T.as_type, () [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%E1.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.9a587c.2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %E2.ref [concrete = constants.%E2.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %E2.ref: type = name_ref E2, file.%E2.decl [concrete = constants.%E2.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_7.2: %E2.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.f45530.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc9_7.2: %E2.type) {
// CHECK:STDOUT:   %T.loc9_7.1: %E2.type = symbolic_binding T, 0 [symbolic = %T.loc9_7.1 (constants.%T.f45530.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc10_6.2: type = facet_access_type %T.loc9_7.1 [symbolic = %T.as_type.loc10_6.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %E1.facet.loc10_6.2: %E1.type = facet_value %T.as_type.loc10_6.2, () [symbolic = %E1.facet.loc10_6.2 (constants.%E1.facet)]
// CHECK:STDOUT:   %F.specific_fn.loc10_3.2: <specific function> = specific_function constants.%F, @F(%E1.facet.loc10_6.2) [symbolic = %F.specific_fn.loc10_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %E2.type = name_ref T, %T.loc9_7.2 [symbolic = %T.loc9_7.1 (constants.%T.f45530.2)]
// CHECK:STDOUT:     %T.as_type.loc10_6.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_6.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %E1.facet.loc10_6.1: %E1.type = facet_value %T.as_type.loc10_6.1, () [symbolic = %E1.facet.loc10_6.2 (constants.%E1.facet)]
// CHECK:STDOUT:     %.loc10: %E1.type = converted %T.ref, %E1.facet.loc10_6.1 [symbolic = %E1.facet.loc10_6.2 (constants.%E1.facet)]
// CHECK:STDOUT:     %F.specific_fn.loc10_3.1: <specific function> = specific_function %F.ref, @F(constants.%E1.facet) [symbolic = %F.specific_fn.loc10_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc10_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.f45530.2) {
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T.f45530.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
