// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon

// --- fail_if_true.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_if_true.carbon:[[@LINE+12]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_if_true.carbon:[[@LINE+8]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:                     ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_if_true.carbon:[[@LINE+4]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var x: () = if true then () else ();

// --- fail_if_false.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_if_false.carbon:[[@LINE+12]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_if_false.carbon:[[@LINE+8]]:23: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:                       ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_if_false.carbon:[[@LINE+4]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var x2: () = if false then () else ();

// --- fail_true_or_false.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_true_or_false.carbon:[[@LINE+8]]:15: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_true_or_false.carbon:[[@LINE+4]]:15: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~~~~~~~
// CHECK:STDERR:
var y: bool = true or false;

// --- fail_false_or_true.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_false_or_true.carbon:[[@LINE+8]]:16: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_false_or_true.carbon:[[@LINE+4]]:16: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~~~~~~
// CHECK:STDERR:
var y2: bool = false or true;

// CHECK:STDOUT: --- fail_if_true.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_if_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_true_or_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_false_or_true.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
