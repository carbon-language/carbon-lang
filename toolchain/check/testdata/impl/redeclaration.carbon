// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/redeclaration.carbon

// --- redeclaration_no_class.carbon
library "[[@TEST_NAME]]";

interface Z {}

impl {} as Z;
impl {} as Z {}

// --- redeclaration_in_class.carbon
library "[[@TEST_NAME]]";

interface Z {}

// `impl as Z` is normalized to `impl Self as Z` so they are treated as
// syntactically the same.
class C {
  impl as Z;
  impl as Z;
  impl Self as Z {}
}

class D {
  impl Self as Z;
  impl Self as Z;
  impl as Z {}
}

// --- fail_todo_redeclaration_reenter_class.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;

class C {
  impl as Z;
}

// TODO: This should typecheck (see #5366).
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class.carbon:[[@LINE+19]]:9
// CHECK:STDERR: ->     │ impl C.(as Z) {}
// CHECK:STDERR:        ·         ┳
// CHECK:STDERR:        ·         ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class.carbon:[[@LINE+13]]:9
// CHECK:STDERR: ->     │ impl C.(as Z) {}
// CHECK:STDERR:        ·         ━┳
// CHECK:STDERR:        ·          ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·             ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class.carbon:[[@LINE+6]]:14
// CHECK:STDERR: ->     │ impl C.(as Z) {}
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C.(as Z) {}

// --- fail_todo_redeclaration_reenter_class_explicit_self.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;

class C {
  impl Self as Z;
}

// TODO: This should typecheck (see #5366).
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_explicit_self.carbon:[[@LINE+20]]:9
// CHECK:STDERR: ->     │ impl C.(Self as Z) {}
// CHECK:STDERR:        ·         ━━┳━
// CHECK:STDERR:        ·           ┗━━┫ error: name `Self` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_explicit_self.carbon:[[@LINE+14]]:19
// CHECK:STDERR: ->     │ impl C.(Self as Z) {}
// CHECK:STDERR:        ·                   ┳
// CHECK:STDERR:        ·                   ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                      ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_explicit_self.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl C.(Self as Z) {}
// CHECK:STDERR:        · ━━━━━━━━━━┳━━━━━━━━━━
// CHECK:STDERR:        ·           ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·              ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C.(Self as Z) {}

// --- fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  impl as Z;
}

// TODO: This should typecheck (see #5366).
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon:[[@LINE+20]]:9
// CHECK:STDERR: ->     │ impl C.(Self as Z) {}
// CHECK:STDERR:        ·         ━━┳━
// CHECK:STDERR:        ·           ┗━━┫ error: name `Self` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon:[[@LINE+14]]:19
// CHECK:STDERR: ->     │ impl C.(Self as Z) {}
// CHECK:STDERR:        ·                   ┳
// CHECK:STDERR:        ·                   ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                      ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ impl C.(Self as Z) {}
// CHECK:STDERR:        · ━━━━━━━━━━┳━━━━━━━━━━
// CHECK:STDERR:        ·           ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·              ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C.(Self as Z) {}

// --- fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  impl Self as Z;
}

// TODO: This should typecheck (see #5366).
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon:[[@LINE+19]]:9
// CHECK:STDERR: ->     │ impl C.(as Z) {}
// CHECK:STDERR:        ·         ┳
// CHECK:STDERR:        ·         ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon:[[@LINE+13]]:9
// CHECK:STDERR: ->     │ impl C.(as Z) {}
// CHECK:STDERR:        ·         ━┳
// CHECK:STDERR:        ·          ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·             ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon:[[@LINE+6]]:14
// CHECK:STDERR: ->     │ impl C.(as Z) {}
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C.(as Z) {}

// --- fail_redeclaration_inside_class_is_separate_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;
impl C as Z;

class C {
  // CHECK:STDERR:        ╭─┤ fail_redeclaration_inside_class_is_separate_impl.carbon:[[@LINE+12]]:3
  // CHECK:STDERR: ->     │   impl C as Z {}
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: found non-final `impl` with the same type structure as another
  // CHECK:STDERR:        ·      ┃        non-final `impl`
  // CHECK:STDERR:        ·      ┃        [ImplNonFinalSameTypeStructure]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        │ impl C as Z {}
  // CHECK:STDERR:        · ──────┬──────
  // CHECK:STDERR:        ·       ╰──┤ other `impl` here [ImplNonFinalSameTypeStructureNote]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl C as Z {}
}

impl C as Z {}

// --- fail_redeclaration_outside_class_is_separate_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  impl C as Z {}
}

// CHECK:STDERR:        ╭─┤ fail_redeclaration_outside_class_is_separate_impl.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │   impl C as Z {}
// CHECK:STDERR:        ·   ──────┬──────
// CHECK:STDERR:        ·         ╰──┤ other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl C as Z {}
// CHECK:STDERR:        · ┳━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: found non-final `impl` with the same type structure as another
// CHECK:STDERR:        ·    ┃        non-final `impl`
// CHECK:STDERR:        ·    ┃        [ImplNonFinalSameTypeStructure]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C as Z {}

// --- fail_redeclaration_in_block_scope_is_separate_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

fn F() {
  musteval fn G() -> auto {
    class C {}
    impl C as Z {}
    return C;
  }
  // This is not a redeclaration, since it's in a different scope.
  // CHECK:STDERR:        ╭─┤ fail_redeclaration_in_block_scope_is_separate_impl.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │     impl C as Z {}
  // CHECK:STDERR:        ·     ──────┬──────
  // CHECK:STDERR:        ·           ╰──┤ other `impl` here [ImplNonFinalSameTypeStructureNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   impl G() as Z {}
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: found non-final `impl` with the same type structure as another
  // CHECK:STDERR:        ·      ┃        non-final `impl`
  // CHECK:STDERR:        ·      ┃        [ImplNonFinalSameTypeStructure]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl G() as Z {}
}

// --- redecl_of_imported_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}
class C {}
impl C as Z {}

// --- fail_redecl_of_imported_impl.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_redecl_of_imported_impl.impl.carbon:[[@LINE+6]]:1
// CHECK:STDERR: ->     │ impl C as Z {}
// CHECK:STDERR:        · ━━━━━━┳━━━━━━
// CHECK:STDERR:        ·       ┗━━┫ error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl C as Z {}

// --- decl_of_imported_impl_in_new_scope.carbon
library "[[@TEST_NAME]]";

interface Z {}
class C {}
impl C as Z {}

// --- fail_decl_of_imported_impl_in_new_scope.impl.carbon
impl library "[[@TEST_NAME]]";

class D {
  // This is not treated as a redeclaration since it is in a different scope.
  // But that means it overlaps fully with the impl from the api file, which is
  // a separate error.

  // CHECK:STDERR:        ╭─┤ fail_decl_of_imported_impl_in_new_scope.impl.carbon:[[@LINE+13]]:3
  // CHECK:STDERR: ->     │   impl C as Z {}
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: found non-final `impl` with the same type structure as another
  // CHECK:STDERR:        ·      ┃        non-final `impl`
  // CHECK:STDERR:        ·      ┃        [ImplNonFinalSameTypeStructure]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ decl_of_imported_impl_in_new_scope.carbon
  // CHECK:STDERR:        │ impl C as Z {}
  // CHECK:STDERR:        · ──────┬──────
  // CHECK:STDERR:        ·       ╰──┤ other `impl` here [ImplNonFinalSameTypeStructureNote]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl C as Z {}
}

// --- fail_redecl_in_fn_scope.carbon
library "[[@TEST_NAME]]";

interface Z {}

fn F() {
  class C {}
  impl C as Z;
  // CHECK:STDERR:        ╭─┤ fail_redecl_in_fn_scope.carbon:[[@LINE+14]]:3
  // CHECK:STDERR: ->     │   impl C as Z {}
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: impl redeclaration not in a declarative scope; redeclaration is
  // CHECK:STDERR:        ·      ┃        allowed only in a class or namespace
  // CHECK:STDERR:        ·      ┃        [ImplDeclInInvalidScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_redecl_in_fn_scope.carbon:[[@LINE-9]]:3
  // CHECK:STDERR: ->     │   impl C as Z;
  // CHECK:STDERR:        ·   ━━━━━━┳━━━━━
  // CHECK:STDERR:        ·         ┗━━┫ error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl C as Z {}
}

// --- fail_redecl_in_different_fn_scope.carbon
library "[[@TEST_NAME]]";

interface Z {}
class C {}

fn F() {
  impl C as Z;
}

fn G() {
  // CHECK:STDERR:        ╭─┤ fail_redecl_in_different_fn_scope.carbon:[[@LINE+14]]:3
  // CHECK:STDERR: ->     │   impl C as Z {}
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: impl redeclaration not in a declarative scope; redeclaration is
  // CHECK:STDERR:        ·      ┃        allowed only in a class or namespace
  // CHECK:STDERR:        ·      ┃        [ImplDeclInInvalidScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_redecl_in_different_fn_scope.carbon:[[@LINE-12]]:3
  // CHECK:STDERR: ->     │   impl C as Z;
  // CHECK:STDERR:        ·   ━━━━━━┳━━━━━
  // CHECK:STDERR:        ·         ┗━━┫ error: impl declared but not defined [ImplMissingDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  impl C as Z {}
}

// --- fail_redecl_in_block_scope.carbon
library "[[@TEST_NAME]]";

interface Z {}

fn F() {
  if (true) {
    class C {}
    impl C as Z;
    // CHECK:STDERR:        ╭─┤ fail_redecl_in_block_scope.carbon:[[@LINE+14]]:5
    // CHECK:STDERR: ->     │     impl C as Z {}
    // CHECK:STDERR:        ·     ┳━━━━━━━━━━━━
    // CHECK:STDERR:        ·     ┗━━┫ error: impl redeclaration not in a declarative scope; redeclaration is
    // CHECK:STDERR:        ·        ┃        allowed only in a class or namespace
    // CHECK:STDERR:        ·        ┃        [ImplDeclInInvalidScope]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    // CHECK:STDERR:        ╭─┤ fail_redecl_in_block_scope.carbon:[[@LINE-9]]:5
    // CHECK:STDERR: ->     │     impl C as Z;
    // CHECK:STDERR:        ·     ━━━━━━┳━━━━━
    // CHECK:STDERR:        ·           ┗━━┫ error: impl declared but not defined [ImplMissingDefinition]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    impl C as Z {}
  }
}

// --- fail_redecl_in_different_block_scope.carbon
library "[[@TEST_NAME]]";

interface Z {}
class C {}

fn F() {
  if (true) {
    impl C as Z;
  }
  if (true) {
    // CHECK:STDERR:        ╭─┤ fail_redecl_in_different_block_scope.carbon:[[@LINE+14]]:5
    // CHECK:STDERR: ->     │     impl C as Z {}
    // CHECK:STDERR:        ·     ┳━━━━━━━━━━━━
    // CHECK:STDERR:        ·     ┗━━┫ error: impl redeclaration not in a declarative scope; redeclaration is
    // CHECK:STDERR:        ·        ┃        allowed only in a class or namespace
    // CHECK:STDERR:        ·        ┃        [ImplDeclInInvalidScope]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    // CHECK:STDERR:        ╭─┤ fail_redecl_in_different_block_scope.carbon:[[@LINE-11]]:5
    // CHECK:STDERR: ->     │     impl C as Z;
    // CHECK:STDERR:        ·     ━━━━━━┳━━━━━
    // CHECK:STDERR:        ·           ┗━━┫ error: impl declared but not defined [ImplMissingDefinition]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    impl C as Z {}
  }
}
