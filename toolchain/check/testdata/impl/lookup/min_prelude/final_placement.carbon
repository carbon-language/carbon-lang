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

// --- interface_z.carbon
library "[[@TEST_NAME]]";

interface Z(T:! type) {
  let X:! type;
}

// Blanket final impl on Z is allowed in the same file.
final impl forall [T:! type] T as Z(T) where .X = {.z_t: ()} {}

// Blanket final impl on Z with a concrete parameter.
final impl forall [T:! type] T as Z(()) where .X = {.z_tuple: ()} {}

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

// --- type_d.carbon
library "[[@TEST_NAME]]";

class D {}

// --- todo_fail_final_with_root_self_different_file.carbon
library "[[@TEST_NAME]]";

import library "type_d";
import library "interface_z";

// TODO: Can not write a final impl on `D` outside the file where `D` is defined.
final impl D as Z({}) where .X = () {}
