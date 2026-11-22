// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_modifiers.carbon

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ protected var b: () = ();
// CHECK:STDERR:        · ━━━━┳━━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `protected` not allowed; requires class scope
// CHECK:STDERR:        ·        ┃        [ModifierProtectedNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
protected var b: () = ();

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+8]]:9
// CHECK:STDERR: ->     │ private protected var c: () = ();
// CHECK:STDERR:        · ───┬─── ━━━━┳━━━━
// CHECK:STDERR:        ·    │        ┗━━┫ error: `protected` not allowed on declaration with `private`
// CHECK:STDERR:        ·    │           ┃        [ModifierNotAllowedWith]
// CHECK:STDERR:        ·    ╰─────────────┤ `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
private protected var c: () = ();

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+14]]:11
// CHECK:STDERR: ->     │ protected protected var d: () = ();
// CHECK:STDERR:        · ────┬──── ━━━━┳━━━━
// CHECK:STDERR:        ·     │         ┗━━┫ error: `protected` repeated on declaration [ModifierRepeated]
// CHECK:STDERR:        ·     ╰──────────────┤ `protected` previously appeared here [ModifierPrevious]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ protected protected var d: () = ();
// CHECK:STDERR:        · ━━━━┳━━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `protected` not allowed; requires class scope
// CHECK:STDERR:        ·        ┃        [ModifierProtectedNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
protected protected var d: () = ();

// CHECK:STDERR:        ╭─┤ fail_modifiers.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ abstract var e: () = ();
// CHECK:STDERR:        · ━━━━┳━━━
// CHECK:STDERR:        ·     ┗━━┫ error: `abstract` not allowed on `var` declaration
// CHECK:STDERR:        ·        ┃        [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
abstract var e: () = ();
