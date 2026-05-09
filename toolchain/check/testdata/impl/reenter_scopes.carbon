// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/reenter_scopes.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/reenter_scopes.carbon

// --- fail_todo_reenter_class_scope.carbon
library "[[@TEST_NAME]]";

namespace N;
interface N.Z { fn ZF(); }
class N.C;

class N.C {
  class D {}
}

// CHECK:STDERR: fail_todo_reenter_class_scope.carbon:[[@LINE+12]]:11: error: expected expression [ExpectedExpr]
// CHECK:STDERR: impl N.C.(as Z) {
// CHECK:STDERR:           ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope.carbon:[[@LINE+8]]:11: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl N.C.(as Z) {
// CHECK:STDERR:           ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope.carbon:[[@LINE+4]]:17: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl N.C.(as Z) {
// CHECK:STDERR:                 ^
// CHECK:STDERR:
impl N.C.(as Z) {
  fn ZF() {
    // D's name is in avaiable since we're inside the scope of `N.C`.
    {} as D;
  }
}

// --- fail_todo_reenter_class_scope_explicit_self.carbon
library "[[@TEST_NAME]]";

namespace N;
interface N.Z { fn ZF(); }
class N.C;

class N.C {
  class D {}
}

// CHECK:STDERR: fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+16]]:11: error: name `Self` not found [NameNotFound]
// CHECK:STDERR: impl N.C.(Self as Z) {
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+12]]:19: error: name `Z` not found [NameNotFound]
// CHECK:STDERR: impl N.C.(Self as Z) {
// CHECK:STDERR:                   ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+8]]:22: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl N.C.(Self as Z) {
// CHECK:STDERR:                      ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl N.C.(Self as Z) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl N.C.(Self as Z) {
  fn ZF() {
    // D's name is in avaiable since we're inside the scope of `N.C`.
    {} as D;
  }
}

// --- fail_todo_reenter_class_scope_unrelated_self.carbon
library "[[@TEST_NAME]]";

namespace N;
interface N.Z { fn ZF(); }
class N.C;

class N.C {
  class D {}
}

// CHECK:STDERR: fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+16]]:11: error: name `D` not found [NameNotFound]
// CHECK:STDERR: impl N.C.(D as Z) {
// CHECK:STDERR:           ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+12]]:16: error: name `Z` not found [NameNotFound]
// CHECK:STDERR: impl N.C.(D as Z) {
// CHECK:STDERR:                ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+8]]:19: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl N.C.(D as Z) {
// CHECK:STDERR:                   ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl N.C.(D as Z) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl N.C.(D as Z) {
  fn ZF() {
    // D's name is in avaiable since we're inside the scope of `N.C`.
    {} as D;
  }
}

// --- fail_todo_reenter_namespace_scope.carbon
library "[[@TEST_NAME]]";

namespace N;
interface N.Z { fn ZF(); }
class N.C {}
class N.D {}

// The path outside the parens must point to a class.
// CHECK:STDERR: fail_todo_reenter_namespace_scope.carbon:[[@LINE+16]]:9: error: name `C` not found [NameNotFound]
// CHECK:STDERR: impl N.(C as Z) {
// CHECK:STDERR:         ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_namespace_scope.carbon:[[@LINE+12]]:14: error: name `Z` not found [NameNotFound]
// CHECK:STDERR: impl N.(C as Z) {
// CHECK:STDERR:              ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_namespace_scope.carbon:[[@LINE+8]]:17: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl N.(C as Z) {
// CHECK:STDERR:                 ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_reenter_namespace_scope.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl N.(C as Z) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl N.(C as Z) {
  fn ZF() {
    // D's name is in avaiable since we're inside the scope of `N`.
    {} as D;
  }
}
