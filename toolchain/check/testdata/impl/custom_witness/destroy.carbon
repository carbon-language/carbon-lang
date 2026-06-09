// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/destroy.carbon

// --- destroy_type.carbon
library "[[@TEST_NAME]]";

fn F() {
  // This looks for a Destroy witness for a TypeType.
  type as Core.Destroy;
}

// --- destroy_facet_type.carbon
library "[[@TEST_NAME]]";

interface Z {}

fn F() {
  // This looks for a Destroy witness for a FacetType.
  (Z & Core.Destroy) as Core.Destroy;
}

// --- destroy_int_literal_type.carbon
library "[[@TEST_NAME]]";

fn F() {
  // This looks for a Destroy witness for an IntLiteralType.
  Core.IntLiteral as Core.Destroy;
}

// --- destroy_int_type.carbon
library "[[@TEST_NAME]]";

fn F() {
  // This looks for a Destroy witness for a class type that adapts IntType, and
  // has a symbolic CompleteTypeWitness.
  Core.Int(32) as Core.Destroy;
}

// --- destroy_symbolic_int_type.carbon
library "[[@TEST_NAME]]";

fn F(N:! Core.IntLiteral) {
  // This looks for a Destroy witness for a class type that adapts IntType with a
  // symbolic type.
  Core.Int(N) as Core.Destroy;
}

// --- destroy_class_type.carbon
library "[[@TEST_NAME]]";

class C(T:! type) {}

fn F() {
  // This looks for a Destroy witness for a class type.
  C({}) as Core.Destroy;
}

// --- fail_destroy_incomplete_class.carbon
library "[[@TEST_NAME]]";

class C(T:! type);

fn F() {
  // This looks for a Destroy witness for an incomplete class type.
  // CHECK:STDERR: fail_destroy_incomplete_class.carbon:[[@LINE+4]]:3: error: cannot convert type `C({})` into type implementing `Core.Destroy` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   C({}) as Core.Destroy;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  C({}) as Core.Destroy;
}
