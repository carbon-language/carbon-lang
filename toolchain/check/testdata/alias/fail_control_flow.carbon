// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+11]]:11: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+7]]:11: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+3]]:11: ERROR: Alias initializer must be a name reference.
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~~~~~~~
alias a = true or false;

// CHECK:STDOUT: --- fail_control_flow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc18: bool = block_arg <unexpected instblockref block4>
// CHECK:STDOUT:   %a: <error> = bind_alias a, <error> [template = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
