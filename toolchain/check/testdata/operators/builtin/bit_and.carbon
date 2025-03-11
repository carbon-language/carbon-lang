// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/bit_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/bit_and.carbon

// --- bit_and_facet_types.carbon
library "[[@TEST_NAME]]";

interface I {}
interface J {}

fn F1[T:! I & J](t: T) {}
fn F2[T:! I & I](t: T) {}

// --- fail_bit_and_non_facet_types.carbon
library "[[@TEST_NAME]]";

interface I {}
class J {}
class K {}

// CHECK:STDERR: fail_bit_and_non_facet_types.carbon:[[@LINE+4]]:11: error: non-facet type `J` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn F1[T:! I & J](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn F1[T:! I & J](t: T) {}
// CHECK:STDERR: fail_bit_and_non_facet_types.carbon:[[@LINE+4]]:11: error: non-facet type `J` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn F2[T:! J & I](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn F2[T:! J & I](t: T) {}
// CHECK:STDERR: fail_bit_and_non_facet_types.carbon:[[@LINE+8]]:11: error: non-facet type `J` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn F3[T:! J & K](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bit_and_non_facet_types.carbon:[[@LINE+4]]:11: error: non-facet type `K` combined with `&` operator [FacetTypeRequiredForTypeAndOperator]
// CHECK:STDERR: fn F3[T:! J & K](t: T) {}
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
fn F3[T:! J & K](t: T) {}

// --- bit_and_values_with_impl.carbon
library "[[@TEST_NAME]]";

class J { adapt {}; }
impl J as Core.BitAnd {
  fn Op[self: Self](other: Self) -> Self { return {} as J; }
}

fn F() {
  let a: J = {} as J;
  let b: J = {} as J;
  a & b;
}

// --- fail_bit_and_values_no_impl.carbon
library "[[@TEST_NAME]]";

class J { adapt {}; }

fn F() {
  let a: J = {} as J;
  let b: J = {} as J;
  // CHECK:STDERR: fail_bit_and_values_no_impl.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.BitAnd` in type `J` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a & b;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  a & b;
}
