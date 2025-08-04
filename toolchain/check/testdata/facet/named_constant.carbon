// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/named_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/named_constant.carbon

// --- fail_todo_named_constant_in_rewrite.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn Test() {
  let Constant:! type = ();

  fn F(T:! I where .X = Constant) {}

  fn G(T:! I where .X = Constant) {
    // TODO: The facet type T in G should match the facet type T in F.
    // CHECK:STDERR: fail_todo_named_constant_in_rewrite.carbon:[[@LINE+7]]:5: error: cannot deduce value for generic parameter `Constant` [DeductionIncomplete]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_todo_named_constant_in_rewrite.carbon:[[@LINE-7]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! I where .X = Constant) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    F(T);
  }
}

// --- fail_todo_named_constant_two_levels_in_rewrite.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn Test() {
  let Constant2:! type = ();
  let Constant:! type = (Constant2, );

  fn F(T:! I where .X = Constant) {}

  fn G(T:! I where .X = Constant) {
    // TODO: The facet type T in G should match the facet type T in F.
    // CHECK:STDERR: fail_todo_named_constant_two_levels_in_rewrite.carbon:[[@LINE+7]]:5: error: cannot deduce value for generic parameter `Constant2` [DeductionIncomplete]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_todo_named_constant_two_levels_in_rewrite.carbon:[[@LINE-7]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! I where .X = Constant) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    F(T);
  }
}

// --- fail_todo_named_constant_in_rewrite_callee.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn Test() {
  let Constant:! type = ();

  fn F(T:! I where .X = Constant) {}

  fn G(T:! I where .X = ()) {
    // TODO: The facet type T in G should match the facet type T in F.
    // CHECK:STDERR: fail_todo_named_constant_in_rewrite_callee.carbon:[[@LINE+7]]:5: error: cannot deduce value for generic parameter `Constant` [DeductionIncomplete]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_todo_named_constant_in_rewrite_callee.carbon:[[@LINE-7]]:3: note: while deducing parameters of generic declared here [DeductionGenericHere]
    // CHECK:STDERR:   fn F(T:! I where .X = Constant) {}
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    F(T);
  }
}

// --- fail_todo_named_constant_in_rewrite_caller.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn Test() {
  let Constant:! type = ();

  fn F(T:! I where .X = ()) {}

  fn G(T:! I where .X = Constant) {
    // TODO: The facet type T in G should match the facet type T in F.
    // CHECK:STDERR: fail_todo_named_constant_in_rewrite_caller.carbon:[[@LINE+7]]:5: error: cannot convert type `T` that implements `I where .(I.X) = Constant` into type implementing `I where .(I.X) = ()` [ConversionFailureFacetToFacet]
    // CHECK:STDERR:     F(T);
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR: fail_todo_named_constant_in_rewrite_caller.carbon:[[@LINE-7]]:8: note: initializing generic parameter `T` declared here [InitializingGenericParam]
    // CHECK:STDERR:   fn F(T:! I where .X = ()) {}
    // CHECK:STDERR:        ^
    // CHECK:STDERR:
    F(T);
  }
}

// --- fail_todo_named_constant_in_nested_facet_type.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn Test() {
  let Constant:! type = ();

  // TODO: The .Self reference in each .X should be different, there should be
  // no cycle here.
  // CHECK:STDERR: fail_todo_named_constant_in_nested_facet_type.carbon:[[@LINE+4]]:12: error: found cycle in facet type constraint for `.(I.X)` [FacetTypeConstraintCycle]
  // CHECK:STDERR:   fn F(T:! I where .X = (I where .X = Constant)) {}
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F(T:! I where .X = (I where .X = Constant)) {}

  // CHECK:STDERR: fail_todo_named_constant_in_nested_facet_type.carbon:[[@LINE+4]]:12: error: found cycle in facet type constraint for `.(I.X)` [FacetTypeConstraintCycle]
  // CHECK:STDERR:   fn G(T:! I where .X = (I where .X = Constant)) {
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn G(T:! I where .X = (I where .X = Constant)) {
    // TODO: The facet type T in G should match the facet type T in F.
    F(T);
  }
}

// --- fail_todo_named_constant_in_nested_facet_type_caller.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn Test() {
  let Constant:! type = ();

  fn F(T:! I where .X = (I where .X = ())) {}

  // TODO: The .Self reference in each .X should be different, there should be
  // no cycle here.
  // CHECK:STDERR: fail_todo_named_constant_in_nested_facet_type_caller.carbon:[[@LINE+4]]:12: error: found cycle in facet type constraint for `.(I.X)` [FacetTypeConstraintCycle]
  // CHECK:STDERR:   fn G(T:! I where .X = (I where .X = Constant)) {
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn G(T:! I where .X = (I where .X = Constant)) {
    // TODO: The facet type T in G should match the facet type T in F.
    F(T);
  }
}

// --- fail_named_constant_in_nested_facet_type_callee.carbon
library "[[@TEST_NAME]]";

interface I {
  let X:! type;
  let Y:! type;
}

fn Test() {
  let Constant:! type = ();

  // TODO: The .Self reference in each .X should be different, there should be
  // no cycle here.
  // CHECK:STDERR: fail_named_constant_in_nested_facet_type_callee.carbon:[[@LINE+4]]:12: error: found cycle in facet type constraint for `.(I.X)` [FacetTypeConstraintCycle]
  // CHECK:STDERR:   fn F(T:! I where .X = (I where .X = Constant)) {}
  // CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn F(T:! I where .X = (I where .X = Constant)) {}

  fn G(T:! I where .X = (I where .X = ())) {
    // TODO: The facet type T in G should match the facet type T in F.
    F(T);
  }
}
