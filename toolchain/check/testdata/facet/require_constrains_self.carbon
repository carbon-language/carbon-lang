// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/require_constrains_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/require_constrains_self.carbon

// --- fail_self_doesnt_meet_constraint.carbon
library "[[@TEST_NAME]]";

interface Y2 {}
interface Y(T:! Y2) {}
// CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE+7]]:37: error: cannot convert type `Self` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
// CHECK:STDERR: constraint W { extend require impls Y(Self); }
// CHECK:STDERR:                                     ^~~~~~~
// CHECK:STDERR: fail_self_doesnt_meet_constraint.carbon:[[@LINE-4]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
// CHECK:STDERR: interface Y(T:! Y2) {}
// CHECK:STDERR:             ^
// CHECK:STDERR:
constraint W { extend require impls Y(Self); }

// --- self_meets_constraint_in_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Y2 {}
interface Y(T:! Y2) {}
constraint W {
  require impls Y2;
  extend require impls Y(Self);
}

fn F(WW:! W) {
  WW as Y2;
  WW as Y(WW);
}

// --- self_meets_different_constraints_in_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z2 {}
interface Z(T:! Z2) {}

interface Y2 {}
interface Y(T:! Y2) {}

constraint W {
  require impls Y2;
  extend require impls Y(Self);
  require impls Z2;
  extend require impls Z(Self);
}

fn F(WW:! W) {
  WW as Y2;
  WW as Z2;
  WW as Y(WW);
  WW as Z(WW);
}

// --- fail_todo_self_meets_constraint_in_interface.carbon
library "[[@TEST_NAME]]";

interface Y2 {}
interface Y(T:! Y2) {}
interface W {
  require impls Y2;
  // TODO: This should typecheck.
  // CHECK:STDERR: fail_todo_self_meets_constraint_in_interface.carbon:[[@LINE+7]]:24: error: cannot convert type `Self` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   extend require impls Y(Self);
  // CHECK:STDERR:                        ^~~~~~~
  // CHECK:STDERR: fail_todo_self_meets_constraint_in_interface.carbon:[[@LINE-7]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Y(T:! Y2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  extend require impls Y(Self);
}

// TODO: These should all typecheck.
fn F(WW:! W) {
  // CHECK:STDERR: fail_todo_self_meets_constraint_in_interface.carbon:[[@LINE+4]]:3: error: cannot convert type `WW` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WW as Y2;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  WW as Y2;
  // CHECK:STDERR: fail_todo_self_meets_constraint_in_interface.carbon:[[@LINE+7]]:9: error: cannot convert type `WW` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WW as Y(WW);
  // CHECK:STDERR:         ^~~~~
  // CHECK:STDERR: fail_todo_self_meets_constraint_in_interface.carbon:[[@LINE-24]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Y(T:! Y2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  WW as Y(WW);
}

// --- fail_todo_elf_meets_different_constraints_in_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z2 {}
interface Z(T:! Z2) {}

interface Y2 {}
interface Y(T:! Y2) {}

interface W {
  require impls Y2;
  // TODO: This should typecheck.
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE+7]]:24: error: cannot convert type `Self` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   extend require impls Y(Self);
  // CHECK:STDERR:                        ^~~~~~~
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE-8]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Y(T:! Y2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  extend require impls Y(Self);
  require impls Z2;
  // TODO: This should typecheck.
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE+7]]:24: error: cannot convert type `Self` that implements `W` into type implementing `Z2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   extend require impls Z(Self);
  // CHECK:STDERR:                        ^~~~~~~
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE-21]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Z(T:! Z2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  extend require impls Z(Self);
}

// TODO: These should all typecheck.
fn F(WW:! W) {
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE+4]]:3: error: cannot convert type `WW` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WW as Y2;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  WW as Y2;
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE+4]]:3: error: cannot convert type `WW` that implements `W` into type implementing `Z2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WW as Z2;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  WW as Z2;
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE+7]]:9: error: cannot convert type `WW` that implements `W` into type implementing `Y2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WW as Y(WW);
  // CHECK:STDERR:         ^~~~~
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE-40]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Y(T:! Y2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  WW as Y(WW);
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE+7]]:9: error: cannot convert type `WW` that implements `W` into type implementing `Z2` [ConversionFailureFacetToFacet]
  // CHECK:STDERR:   WW as Z(WW);
  // CHECK:STDERR:         ^~~~~
  // CHECK:STDERR: fail_todo_elf_meets_different_constraints_in_named_constraint.carbon:[[@LINE-51]]:13: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: interface Z(T:! Z2) {}
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  WW as Z(WW);
}
