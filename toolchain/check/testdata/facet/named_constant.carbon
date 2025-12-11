// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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
  // CHECK:STDERR: fail_todo_named_constant_in_rewrite.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let Constant:! type = ();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let Constant:! type = ();

  fn F(T:! I where .X = Constant) {}

  fn G(T:! I where .X = Constant) {
    // TODO: The facet type T in G should match the facet type T in F.
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
  // CHECK:STDERR: fail_todo_named_constant_two_levels_in_rewrite.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let Constant2:! type = ();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let Constant2:! type = ();
  let Constant:! type = (Constant2, );

  fn F(T:! I where .X = Constant) {}

  fn G(T:! I where .X = Constant) {
    // TODO: The facet type T in G should match the facet type T in F.
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
  // CHECK:STDERR: fail_todo_named_constant_in_rewrite_callee.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let Constant:! type = ();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let Constant:! type = ();

  fn F(T:! I where .X = Constant) {}

  fn G(T:! I where .X = ()) {
    // TODO: The facet type T in G should match the facet type T in F.
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
  // CHECK:STDERR: fail_todo_named_constant_in_rewrite_caller.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let Constant:! type = ();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let Constant:! type = ();

  fn F(T:! I where .X = ()) {}

  fn G(T:! I where .X = Constant) {
    // TODO: The facet type T in G should match the facet type T in F.
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
  // CHECK:STDERR: fail_todo_named_constant_in_nested_facet_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let Constant:! type = ();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let Constant:! type = ();

  // TODO: The .Self reference in each .X should be different, there should be
  // no cycle here.
  fn F(T:! I where .X = (I where .X = Constant)) {}

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
  // CHECK:STDERR: fail_todo_named_constant_in_nested_facet_type_caller.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let Constant:! type = ();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let Constant:! type = ();

  fn F(T:! I where .X = (I where .X = ())) {}

  // TODO: The .Self reference in each .X should be different, there should be
  // no cycle here.
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
  // CHECK:STDERR: fail_named_constant_in_nested_facet_type_callee.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let Constant:! type = ();
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let Constant:! type = ();

  // TODO: The .Self reference in each .X should be different, there should be
  // no cycle here.
  fn F(T:! I where .X = (I where .X = Constant)) {}

  fn G(T:! I where .X = (I where .X = ())) {
    // TODO: The facet type T in G should match the facet type T in F.
    F(T);
  }
}
