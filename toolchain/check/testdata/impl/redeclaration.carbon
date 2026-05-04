// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
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

// TODO: This should typecheck (see p5366).
// CHECK:STDERR: fail_todo_redeclaration_reenter_class.carbon:[[@LINE+12]]:9: error: expected expression [ExpectedExpr]
// CHECK:STDERR: impl C.(as Z) {}
// CHECK:STDERR:         ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class.carbon:[[@LINE+8]]:9: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl C.(as Z) {}
// CHECK:STDERR:         ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class.carbon:[[@LINE+4]]:15: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl C.(as Z) {}
// CHECK:STDERR:               ^
// CHECK:STDERR:
impl C.(as Z) {}

// --- fail_todo_redeclaration_reenter_class_explicit_self.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;

class C {
  impl Self as Z;
}

// TODO: This should typecheck (see p5366).
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_explicit_self.carbon:[[@LINE+12]]:9: error: name `Self` not found [NameNotFound]
// CHECK:STDERR: impl C.(Self as Z) {}
// CHECK:STDERR:         ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_explicit_self.carbon:[[@LINE+8]]:20: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl C.(Self as Z) {}
// CHECK:STDERR:                    ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_explicit_self.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl C.(Self as Z) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C.(Self as Z) {}

// --- fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  impl as Z;
}

// TODO: This should typecheck (see p5366).
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon:[[@LINE+12]]:9: error: name `Self` not found [NameNotFound]
// CHECK:STDERR: impl C.(Self as Z) {}
// CHECK:STDERR:         ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon:[[@LINE+8]]:20: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl C.(Self as Z) {}
// CHECK:STDERR:                    ^
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_mixed_self_outside.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl C.(Self as Z) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C.(Self as Z) {}

// --- fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  impl Self as Z;
}

// TODO: This should typecheck (see p5366).
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon:[[@LINE+12]]:9: error: expected expression [ExpectedExpr]
// CHECK:STDERR: impl C.(as Z) {}
// CHECK:STDERR:         ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon:[[@LINE+8]]:9: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: impl C.(as Z) {}
// CHECK:STDERR:         ^~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_redeclaration_reenter_class_mixed_self_inside.carbon:[[@LINE+4]]:15: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl C.(as Z) {}
// CHECK:STDERR:               ^
// CHECK:STDERR:
impl C.(as Z) {}

// --- fail_redeclaration_inside_class_is_separate_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C;
impl C as Z;

class C {
  // CHECK:STDERR: fail_redeclaration_inside_class_is_separate_impl.carbon:[[@LINE+3]]:3: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
  // CHECK:STDERR:   impl C as Z {}
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  impl C as Z {}
}

// CHECK:STDERR: fail_redeclaration_inside_class_is_separate_impl.carbon:[[@LINE+4]]:1: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR: impl C as Z {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z {}

// --- fail_redeclaration_outside_class_is_separate_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

class C {
  impl C as Z {}
}

// CHECK:STDERR: fail_redeclaration_outside_class_is_separate_impl.carbon:[[@LINE+7]]:1: error: found non-final `impl` with the same type structure as another non-final `impl` [ImplNonFinalSameTypeStructure]
// CHECK:STDERR: impl C as Z {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redeclaration_outside_class_is_separate_impl.carbon:[[@LINE-6]]:3: note: other `impl` here [ImplNonFinalSameTypeStructureNote]
// CHECK:STDERR:   impl C as Z {}
// CHECK:STDERR:   ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z {}

// --- fail_todo_redeclaration_in_block_scope_is_separate_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}

fn F() {
  // CHECK:STDERR: fail_todo_redeclaration_in_block_scope_is_separate_impl.carbon:[[@LINE+4]]:22: error: semantics TODO: `HandleAutoTypeLiteral` [SemanticsTodo]
  // CHECK:STDERR:   musteval fn G() -> auto {
  // CHECK:STDERR:                      ^~~~
  // CHECK:STDERR:
  musteval fn G() -> auto {
    class C {}
    impl C as Z {}
    return C;
  }
  // TODO: This is not a redeclaration, since it's in a different scope.
  // TODO: found non-final `impl` with the same type structure as another non-final `impl`.
  impl G() as Z {}
}

// --- redecl_of_imported_impl.carbon
library "[[@TEST_NAME]]";

interface Z {}
class C {}
impl C as Z {}

// --- fail_redecl_of_imported_impl.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_redecl_of_imported_impl.impl.carbon:[[@LINE+4]]:1: error: redeclaration of imported impl [RedeclImportedImpl]
// CHECK:STDERR: impl C as Z {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C as Z {}

// --- decl_of_imported_impl_in_new_scope.carbon
library "[[@TEST_NAME]]";

interface Z {}
class C {}
impl C as Z {}

// --- todo_fail_decl_of_imported_impl_in_new_scope.impl.carbon
impl library "[[@TEST_NAME]]";

class D {
  // TODO: This is not treated as a redeclaration since it is in a different
  // scope. But it should be rejected by the orphan rule v2.
  impl C as Z {}
}
