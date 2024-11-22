// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/fail_control_flow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/fail_control_flow.carbon

// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+11]]:11: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+7]]:11: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+3]]:11: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~~~~~~~
alias a = true or false;

// CHECK:STDOUT: --- fail_control_flow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc22: bool = block_arg <unexpected instblockref block5> [template = constants.%.1]
// CHECK:STDOUT:   %a: <error> = bind_alias a, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
