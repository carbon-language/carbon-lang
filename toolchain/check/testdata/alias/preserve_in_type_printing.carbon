// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/array.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/preserve_in_type_printing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/preserve_in_type_printing.carbon

// --- fail_alias_nested_in_type.carbon

library "[[@TEST_NAME]]";

class C {}
alias A = C;

// The diagnostics below should refer to `A`, not `C`.

// CHECK:STDERR:        ╭─┤ fail_alias_nested_in_type.carbon:[[@LINE+7]]:15
// CHECK:STDERR: ->     │ fn F() -> A { return; }
// CHECK:STDERR:        ·           ┬   ━━━┳━━━
// CHECK:STDERR:        ·           │      ┗━━┫ error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR:        ·           ╰───────────┤ return type of function is `A` [ReturnTypeHereNote]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> A { return; }
// CHECK:STDERR:        ╭─┤ fail_alias_nested_in_type.carbon:[[@LINE+7]]:16
// CHECK:STDERR: ->     │ fn G() -> A* { return; }
// CHECK:STDERR:        ·           ─┬   ━━━┳━━━
// CHECK:STDERR:        ·            │      ┗━━┫ error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR:        ·            ╰───────────┤ return type of function is `A*` [ReturnTypeHereNote]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn G() -> A* { return; }
// CHECK:STDERR:        ╭─┤ fail_alias_nested_in_type.carbon:[[@LINE+9]]:22
// CHECK:STDERR: ->     │ fn H() -> const A* { return; }
// CHECK:STDERR:        ·           ────┬───   ━━━┳━━━
// CHECK:STDERR:        ·               │         ┗━━┫ error: missing return value
// CHECK:STDERR:        ·               │            ┃        [ReturnStatementMissingExpr]
// CHECK:STDERR:        ·               ╰──────────────┤ return type of function is `const A*`
// CHECK:STDERR:        ·                              │ [ReturnTypeHereNote]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn H() -> const A* { return; }
// TODO: Preserve type sugar in generic arguments, such as for `Core.Array`.
// CHECK:STDERR:        ╭─┤ fail_alias_nested_in_type.carbon:[[@LINE+9]]:26
// CHECK:STDERR: ->     │ fn I() -> array(A, 42) { return; }
// CHECK:STDERR:        ·           ──────┬─────   ━━━┳━━━
// CHECK:STDERR:        ·                 │           ┗━━┫ error: missing return value
// CHECK:STDERR:        ·                 │              ┃        [ReturnStatementMissingExpr]
// CHECK:STDERR:        ·                 ╰────────────────┤ return type of function is `array(C, 42)`
// CHECK:STDERR:        ·                                  │ [ReturnTypeHereNote]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn I() -> array(A, 42) { return; }

// --- fail_alias_in_type_of_expr.carbon

library "[[@TEST_NAME]]";

class C {}
alias A = C;

fn MakePtr() -> A*;

// The diagnostics below should describe the types of the expressions using the
// spelling `A` from the declaration of the callee, not the canonical `C`.

fn F() {
  // The type of a call is the declared return type of the callee.
  // CHECK:STDERR:        ╭─┤ fail_alias_in_type_of_expr.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   MakePtr().missing;
  // CHECK:STDERR:        ·   ━━━━━━━━┳━━━━━━━━
  // CHECK:STDERR:        ·           ┗━━┫ error: type `A*` does not support qualified expressions
  // CHECK:STDERR:        ·              ┃        [QualifiedExprUnsupported]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  MakePtr().missing;

  // The type of a dereference is the pointee type of the pointer's type.
  // CHECK:STDERR:        ╭─┤ fail_alias_in_type_of_expr.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   (*MakePtr())();
  // CHECK:STDERR:        ·   ━━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: value of type `A` is not callable [CallToNonCallable]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  (*MakePtr())();
}
