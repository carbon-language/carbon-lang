// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon

// --- fail_if_true.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_if_true.carbon:[[@LINE+27]]:13
// CHECK:STDERR: ->     │ var x: () = if true then () else ();
// CHECK:STDERR:        ·             ━━━┳━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_if_true.carbon:[[@LINE+18]]:21
// CHECK:STDERR: ->     │ var x: () = if true then () else ();
// CHECK:STDERR:        ·                     ━━━┳━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_if_true.carbon:[[@LINE+9]]:13
// CHECK:STDERR: ->     │ var x: () = if true then () else ();
// CHECK:STDERR:        ·             ━━━━━━━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: () = if true then () else ();

// --- fail_if_false.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_if_false.carbon:[[@LINE+27]]:14
// CHECK:STDERR: ->     │ var x2: () = if false then () else ();
// CHECK:STDERR:        ·              ━━━━┳━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_if_false.carbon:[[@LINE+18]]:23
// CHECK:STDERR: ->     │ var x2: () = if false then () else ();
// CHECK:STDERR:        ·                       ━━━┳━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_if_false.carbon:[[@LINE+9]]:14
// CHECK:STDERR: ->     │ var x2: () = if false then () else ();
// CHECK:STDERR:        ·              ━━━━━━━━━━━━┳━━━━━━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x2: () = if false then () else ();

// --- fail_true_or_false.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_true_or_false.carbon:[[@LINE+18]]:15
// CHECK:STDERR: ->     │ var y: bool = true or false;
// CHECK:STDERR:        ·               ━━━┳━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_true_or_false.carbon:[[@LINE+9]]:15
// CHECK:STDERR: ->     │ var y: bool = true or false;
// CHECK:STDERR:        ·               ━━━━━━┳━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var y: bool = true or false;

// --- fail_false_or_true.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_false_or_true.carbon:[[@LINE+18]]:16
// CHECK:STDERR: ->     │ var y2: bool = false or true;
// CHECK:STDERR:        ·                ━━━━┳━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_false_or_true.carbon:[[@LINE+9]]:16
// CHECK:STDERR: ->     │ var y2: bool = false or true;
// CHECK:STDERR:        ·                ━━━━━━┳━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: semantics TODO:
// CHECK:STDERR:        · ┃        `Control flow expressions are currently only supported inside functions.`
// CHECK:STDERR:        · ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var y2: bool = false or true;

// CHECK:STDOUT: --- fail_if_true.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_if_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_true_or_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_false_or_true.carbon
// CHECK:STDOUT:
