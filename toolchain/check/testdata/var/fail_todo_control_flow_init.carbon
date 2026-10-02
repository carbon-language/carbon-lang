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

// CHECK:STDERR: fail_if_true.carbon:[[@LINE+6]]:13-19: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_if_true.carbon:[[@LINE+4]]:21-27: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_if_true.carbon:[[@LINE+2]]:13-35: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var x: () = if true then () else ();

// --- fail_if_false.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_if_false.carbon:[[@LINE+6]]:14-21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_if_false.carbon:[[@LINE+4]]:23-29: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_if_false.carbon:[[@LINE+2]]:14-37: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var x2: () = if false then () else ();

// --- fail_true_or_false.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_true_or_false.carbon:[[@LINE+4]]:15-21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_true_or_false.carbon:[[@LINE+2]]:15-27: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
var y: bool = true or false;

// --- fail_false_or_true.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_false_or_true.carbon:[[@LINE+4]]:16-23: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR:
// CHECK:STDERR: fail_false_or_true.carbon:[[@LINE+2]]:16-28: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
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
