// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/generic_method_constraint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/generic_method_constraint.carbon

// Regression tests for https://github.com/carbon-language/carbon-lang/issues/7919:
// building an impl witness thunk must not crash when an associated function's
// generic parameter constraint depends on an enclosing interface parameter.

// --- dependent_on_interface_param.carbon

library "[[@TEST_NAME]]";

interface A(T: type) {
  fn F(self);
}

interface B(T: type) {
  fn G[U: A(T)](u: U);
}

class C {}

impl C as B(()) {
  fn G[U: A(())](unused u: U) {}
}

// --- dependent_on_self.carbon

library "[[@TEST_NAME]]";

interface A(T: type) {
  fn F(self);
}

interface B(T: type) {
  fn G[U: A(Self)](u: U);
}

class C {}

impl C as B(()) {
  fn G[U: A(C)](unused u: U) {}
}

// --- independent_constraint_still_ok.carbon

library "[[@TEST_NAME]]";

interface A {
  fn F(self);
}

interface B(T: type) {
  fn G[U: A](u: U, t: T);
}

class C {}

impl C as B(()) {
  fn G[U: A](unused u: U, unused t: ()) {}
}
