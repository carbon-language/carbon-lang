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

fn F(T:! E) {}

//@dump-sem-ir-begin
fn G(T:! E & E) {
  F(T);
}
//@dump-sem-ir-end

// --- compatible_constraints.carbon
library "[[@TEST_NAME]]";

constraint E1 {}
constraint E2 {}

fn F(T:! E1) {}

//@dump-sem-ir-begin
fn G(T:! E2) {
  F(T);
}
//@dump-sem-ir-end

// --- fail_todo_unspecified.carbon
library "[[@TEST_NAME]]";

interface Z {}

constraint E {
  // CHECK:STDERR: fail_todo_unspecified.carbon:[[@LINE+4]]:3: error: semantics TODO: `require` [SemanticsTodo]
  // CHECK:STDERR:   require impls Z;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  require impls Z;
}

fn F(T:! E) {}

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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %T: %type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e25: type = pattern_type %type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.8a6: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.Op.type.9a3: type = fn_type @BitAndWith.Op, @BitAndWith(type) [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.8a6 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %.fa7: type = fn_type_with_self_type %BitAndWith.Op.type.9a3, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.636: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.636), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e25 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_12.1: type = splice_block %.loc8_12.3 [concrete = constants.%type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %E.ref.loc8_10: type = name_ref E, file.%E.decl [concrete = constants.%type]
// CHECK:STDOUT:       %E.ref.loc8_14: type = name_ref E, file.%E.decl [concrete = constants.%type]
// CHECK:STDOUT:       %impl.elem0: %.fa7 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %E.ref.loc8_10, %impl.elem0 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method(%E.ref.loc8_10, %E.ref.loc8_14) [concrete = constants.%type]
// CHECK:STDOUT:       %.loc8_12.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%type]
// CHECK:STDOUT:       %.loc8_12.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc8_12.2 [concrete = constants.%type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_6.2: %type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc8_6.2: %type) {
// CHECK:STDOUT:   %T.loc8_6.1: %type = symbolic_binding T, 0 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.specific_fn.loc9_3.2: <specific function> = specific_function constants.%F, @F(%T.loc8_6.1) [symbolic = %F.specific_fn.loc9_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %type = name_ref T, %T.loc8_6.2 [symbolic = %T.loc8_6.1 (constants.%T)]
// CHECK:STDOUT:     %F.specific_fn.loc9_3.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc9_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc9_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc8_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compatible_constraints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %T: %type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9: type = splice_block %E2.ref [concrete = constants.%type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %E2.ref: type = name_ref E2, file.%E2.decl [concrete = constants.%type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_6.2: %type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc9_6.2: %type) {
// CHECK:STDOUT:   %T.loc9_6.1: %type = symbolic_binding T, 0 [symbolic = %T.loc9_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.specific_fn.loc10_3.2: <specific function> = specific_function constants.%F, @F(%T.loc9_6.1) [symbolic = %F.specific_fn.loc10_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: %type = name_ref T, %T.loc9_6.2 [symbolic = %T.loc9_6.1 (constants.%T)]
// CHECK:STDOUT:     %F.specific_fn.loc10_3.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc10_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc10_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc9_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
