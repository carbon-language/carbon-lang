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

// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope.carbon:[[@LINE+19]]:11
// CHECK:STDERR: ->     │ impl N.C.(as Z) {
// CHECK:STDERR:        ·           ━┳
// CHECK:STDERR:        ·            ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope.carbon:[[@LINE+13]]:11
// CHECK:STDERR: ->     │ impl N.C.(as Z) {
// CHECK:STDERR:        ·           ━┳
// CHECK:STDERR:        ·            ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·               ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope.carbon:[[@LINE+6]]:17
// CHECK:STDERR: ->     │ impl N.C.(as Z) {
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR:        ╰────
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

// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+26]]:11
// CHECK:STDERR: ->     │ impl N.C.(Self as Z) {
// CHECK:STDERR:        ·           ━━┳━
// CHECK:STDERR:        ·             ┗━━┫ error: name `Self` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+20]]:19
// CHECK:STDERR: ->     │ impl N.C.(Self as Z) {
// CHECK:STDERR:        ·                   ┳
// CHECK:STDERR:        ·                   ┗━━┫ error: name `Z` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+14]]:22
// CHECK:STDERR: ->     │ impl N.C.(Self as Z) {
// CHECK:STDERR:        ·                      ┳
// CHECK:STDERR:        ·                      ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                         ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_explicit_self.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl N.C.(Self as Z) {
// CHECK:STDERR:        · ━━━━━━━━━━━┳━━━━━━━━━━
// CHECK:STDERR:        ·            ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·               ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
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

// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+26]]:11
// CHECK:STDERR: ->     │ impl N.C.(D as Z) {
// CHECK:STDERR:        ·           ┳
// CHECK:STDERR:        ·           ┗━━┫ error: name `D` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+20]]:16
// CHECK:STDERR: ->     │ impl N.C.(D as Z) {
// CHECK:STDERR:        ·                ┳
// CHECK:STDERR:        ·                ┗━━┫ error: name `Z` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+14]]:19
// CHECK:STDERR: ->     │ impl N.C.(D as Z) {
// CHECK:STDERR:        ·                   ┳
// CHECK:STDERR:        ·                   ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                      ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_class_scope_unrelated_self.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl N.C.(D as Z) {
// CHECK:STDERR:        · ━━━━━━━━━┳━━━━━━━━━
// CHECK:STDERR:        ·          ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·             ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
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
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_namespace_scope.carbon:[[@LINE+25]]:9
// CHECK:STDERR: ->     │ impl N.(C as Z) {
// CHECK:STDERR:        ·         ┳
// CHECK:STDERR:        ·         ┗━━┫ error: name `C` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_namespace_scope.carbon:[[@LINE+19]]:14
// CHECK:STDERR: ->     │ impl N.(C as Z) {
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: name `Z` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_namespace_scope.carbon:[[@LINE+13]]:17
// CHECK:STDERR: ->     │ impl N.(C as Z) {
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_reenter_namespace_scope.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl N.(C as Z) {
// CHECK:STDERR:        · ━━━━━━━━┳━━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·            ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl N.(C as Z) {
  fn ZF() {
    // D's name is in avaiable since we're inside the scope of `N`.
    {} as D;
  }
}
