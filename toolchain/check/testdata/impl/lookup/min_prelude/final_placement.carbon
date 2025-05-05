// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/min_prelude/final_placement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/min_prelude/final_placement.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- interface_z.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

// Blanket final impl on Z is allowed in the same file.
final impl forall [T:! type] T as Z(T) where .X = {.z_t: ()} {}

// Blanket final impl on Z with a concrete parameter.
final impl forall [T:! type] T as Z(()) where .X = {.z_tuple: ()} {}

// --- type_d.carbon
library "[[@TEST_NAME]]";

class D(T:! type) {}

// ============================================================================
// Test files
// ============================================================================

// --- todo_final_with_root_self_same_file.carbon
library "[[@TEST_NAME]]";

import library "interface_z";

class C(T:! type) { adapt (); }

// Can provide a specialized final blanket impl for a type defined in the same
// file.
final impl forall [T:! type] C(T) as Z(T) where .X = {.c: ()} {}

fn F() {
  // TODO: It's not yet decided which `final` specialization wins, but it must
  // be one from the interface file.
  //
  // See https://github.com/carbon-language/carbon-lang/pull/5337.
  let x: C(()).(Z(()).X) = {.c = ()};
}

// --- todo_fail_final_with_symbolic_self.carbon
library "[[@TEST_NAME]]";

import library "interface_z";

class C;

// TODO: Can not make a specialized final blanket impl over `Z` for all types
// except in the same file as `Z`, even if a type from the current file appears
// in a parameter of `Z`.
final impl forall [T:! type] T as Z(C) where .X = () {}

// --- todo_fail_final_with_root_self_different_file.carbon
library "[[@TEST_NAME]]";

import library "interface_z";
import library "type_d";

// TODO: Can not write a final impl on `D` outside the file where `D` is defined, unless it's in the same file as the interface.
final impl D(()) as Z({}) where .X = () {}

// --- final_with_root_self_different_file_but_interface_same_file.carbon
library "[[@TEST_NAME]]";

import library "type_d";

interface Y {}

// The final impl is for a root self type from another file, but the interface
// is in the same file, so this is valid.
final impl D(()) as Y {}

// The same thing with a generic parameter, so the self type is symbolic, but
// the root type is from a different file.
final impl forall [T:! type] D(T) as Y {}

class C;

// The same thing with a type from the same file in a parameter, but the root
// tipe is from a different file.
final impl D(C) as Y {}

// --- todo_fail_final_specialized_partially_overlapped_in_same_file_as_interface.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type, U:! type) {
  let X:! type;
}

// Blanket final impl on Z in the same file as Z.
final impl forall [T:! type] T as Z(T, ()) where .X = {.z_t: ()} {}

class C;

// final impl specialized on C for Z, also in the same file as Z.
//
// TODO: It's not fully decided yet, but at the moment, specializations of final
// impls are not chosen when there's a more general impl available, even when it
// is written in the same file.
final impl forall [U:! type] C as Z(C, U) where .X = {.z_c: ()} {}

fn F() {
  // TODO: It's not yet decided which `final` specialization wins, but for now
  // it should be one the blanket impl.
  //
  // See https://github.com/carbon-language/carbon-lang/pull/5337.
  let x: C.(Z(C, ()).X) = {.z_c = ()};
}

// --- fail_todo_final_specialized_fully_overlapped_in_same_file_as_interface.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

// Blanket final impl on Z in the same file as Z.
final impl forall [T:! type] T as Z(T) where .X = {.z_t: ()} {}

class C;

// final impl specialized on C for Z, also in the same file as Z.
//
// TODO: It's not fully decided yet, but at the moment, specializations of final
// impls that are fully overlapped are never selected so should generate an
// error.
final impl C as Z(C) where .X = {.z_c: ()} {}

fn F() {
  // TODO: It's not yet fully decided but at the moment only the blanket impl is
  // considered valid, so it should be selected.
  //
  // CHECK:STDERR: fail_todo_final_specialized_fully_overlapped_in_same_file_as_interface.carbon:[[@LINE+7]]:13: error: 2 arguments passed to generic interface expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   let x: C.(Z(C, ()).X) = {.z_t = ()};
  // CHECK:STDERR:             ^~~~~~~~
  // CHECK:STDERR: fail_todo_final_specialized_fully_overlapped_in_same_file_as_interface.carbon:[[@LINE-23]]:1: note: calling generic interface declared here [InCallToEntity]
  // CHECK:STDERR: interface Z(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let x: C.(Z(C, ()).X) = {.z_t = ()};
}
