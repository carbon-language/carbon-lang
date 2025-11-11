// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/error_recovery.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/error_recovery.carbon

// --- fail_runtime_generic_param.carbon
library "[[@TEST_NAME]]";

class C {}
interface I {}

//@dump-sem-ir-begin
// CHECK:STDERR: fail_runtime_generic_param.carbon:[[@LINE+4]]:14: error: parameters of generic types must be constant [GenericParamMustBeConstant]
// CHECK:STDERR: impl forall [T: type] C as I {}
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
impl forall [T: type] C as I {}
//@dump-sem-ir-end

// --- fail_nonfinal_lookup_impl_witness_error_in_import.carbon
library "[[@TEST_NAME]]";

interface Z {}
// CHECK:STDERR: fail_nonfinal_lookup_impl_witness_error_in_import.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl forall [T:! type] T as Z;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z;

fn F(U:! Z) {}

//@dump-sem-ir-begin
fn G(T:! type) {
  // This makes a LookupImplWitness instruction, but future lookups (evaluation
  // of this instruction with a specific) will result in an error since the impl
  // is never defined and is left with an error as its witness at the end of the
  // file. The lookups should not fail entirely, just result in an error
  // witness.
  F(T);
}
//@dump-sem-ir-end

// --- nonfinal_lookup_impl_witness_error_in_import.impl.carbon
impl library "[[@TEST_NAME]]";

fn H() {
  // The specific here contains errors, but does not fail entirely and crash
  // when resolving the LookupImplWitness.
  G(());
}

// --- fail_nonfinal_lookup_impl_witness_error.carbon
library "[[@TEST_NAME]]";

interface Z {}
// CHECK:STDERR: fail_nonfinal_lookup_impl_witness_error.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl forall [T:! type] T as Z;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z;

fn F(U:! Z) {}

//@dump-sem-ir-begin
fn G(T:! type) {
  // This makes a LookupImplWitness instruction, but future lookups (evaluation
  // of this instruction with a specific) will fail with an error since the impl
  // is never defined and is left with an error as its witness at the end of the
  // file. The lookups should not fail entirely, just result in an error
  // witness.
  F(T);
}
//@dump-sem-ir-end

fn H() {
  // The specific here contains errors, but does not fail entirely and crash
  // when resolving the LookupImplWitness.
  G(());
}

// --- fail_final_lookup_impl_witness_error.carbon
library "[[@TEST_NAME]]";

interface Z {}
// CHECK:STDERR: fail_final_lookup_impl_witness_error.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl forall [T:! type] T as Z;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl forall [T:! type] T as Z;

fn F(U:! Z) {}

fn G() {
  // This impl lookup resolves to a final witness, which poisons any future
  // queries. At the end of the file, the poisoned queries are replayed to make
  // sure they don't change. However, here it is changed by the impl being
  // diagnosed with an error. The poisoning check should handle that gracefully.
  //@dump-sem-ir-begin
  F(());
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_runtime_generic_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness file.%I.impl_witness_table [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @C.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (), @C.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.I.impl: %C.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_lookup_impl_witness_error_in_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %T, (%Z.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Z.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc13_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc13_6.2: type) {
// CHECK:STDOUT:   %T.loc13_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_6.1, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %Z.facet.loc19_6.2: %Z.type = facet_value %T.loc13_6.1, (%Z.lookup_impl_witness) [symbolic = %Z.facet.loc19_6.2 (constants.%Z.facet)]
// CHECK:STDOUT:   %F.specific_fn.loc19_3.2: <specific function> = specific_function constants.%F, @F(%Z.facet.loc19_6.2) [symbolic = %F.specific_fn.loc19_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:     %Z.facet.loc19_6.1: %Z.type = facet_value %T.ref, (constants.%Z.lookup_impl_witness) [symbolic = %Z.facet.loc19_6.2 (constants.%Z.facet)]
// CHECK:STDOUT:     %.loc19: %Z.type = converted %T.ref, %Z.facet.loc19_6.1 [symbolic = %Z.facet.loc19_6.2 (constants.%Z.facet)]
// CHECK:STDOUT:     %F.specific_fn.loc19_3.1: <specific function> = specific_function %F.ref, @F(constants.%Z.facet) [symbolic = %F.specific_fn.loc19_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc19_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nonfinal_lookup_impl_witness_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %T, (%Z.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Z.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %T.loc13_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc13_6.2: type) {
// CHECK:STDOUT:   %T.loc13_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_6.1, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness)]
// CHECK:STDOUT:   %Z.facet.loc19_6.2: %Z.type = facet_value %T.loc13_6.1, (%Z.lookup_impl_witness) [symbolic = %Z.facet.loc19_6.2 (constants.%Z.facet)]
// CHECK:STDOUT:   %F.specific_fn.loc19_3.2: <specific function> = specific_function constants.%F, @F(%Z.facet.loc19_6.2) [symbolic = %F.specific_fn.loc19_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:     %Z.facet.loc19_6.1: %Z.type = facet_value %T.ref, (constants.%Z.lookup_impl_witness) [symbolic = %Z.facet.loc19_6.2 (constants.%Z.facet)]
// CHECK:STDOUT:     %.loc19: %Z.type = converted %T.ref, %Z.facet.loc19_6.1 [symbolic = %Z.facet.loc19_6.2 (constants.%Z.facet)]
// CHECK:STDOUT:     %F.specific_fn.loc19_3.1: <specific function> = specific_function %F.ref, @F(constants.%Z.facet) [symbolic = %F.specific_fn.loc19_3.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc19_3.1()
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Z.lookup_impl_witness => <error>
// CHECK:STDOUT:   %Z.facet.loc19_6.2 => <error>
// CHECK:STDOUT:   %F.specific_fn.loc19_3.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_final_lookup_impl_witness_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Z.impl_witness.b07: <witness> = impl_witness file.%Z.impl_witness_table, @T.as.Z.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %empty_tuple.type, (%Z.impl_witness.b07) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Z.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_6: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value constants.%empty_tuple.type, (constants.%Z.impl_witness.b07) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc18_7: %Z.type = converted %.loc18_6, %Z.facet [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Z.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
