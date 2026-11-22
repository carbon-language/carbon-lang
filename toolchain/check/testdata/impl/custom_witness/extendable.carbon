// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/extendable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/extendable.carbon

// --- base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

fn Test() {
  B as Core.Extendable;
}

// --- abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class A {}

fn Test() {
  A as Core.Extendable;
}

// --- generic_base_class.carbon
library "[[@TEST_NAME]]";

base class B(unused T: type) {}

fn Test() {
  B(()) as Core.Extendable;
}

fn TestSymbolic[T: type]() {
  B(T) as Core.Extendable;
}

// --- fail_class.carbon
library "[[@TEST_NAME]]";

class C {}

fn Test() {
  C as Core.Extendable;
}

// --- fail_derived_from_base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

class C {
  extend base: B;
}

fn Test() {
  C as Core.Extendable;
}

// --- fail_partial_class.carbon
library "[[@TEST_NAME]]";

base class B {}

fn Test() {
  partial B as Core.Extendable;
}

// --- fail_tuple.carbon
library "[[@TEST_NAME]]";

fn Test() {
  ((), ()) as Core.Extendable;
}

// --- fail_struct.carbon
library "[[@TEST_NAME]]";

fn Test() {
  {.a: ()} as Core.Extendable;
}

// --- fail_symbolic.carbon
library "[[@TEST_NAME]]";

fn Test[T: type]() {
  T as Core.Extendable;
}

