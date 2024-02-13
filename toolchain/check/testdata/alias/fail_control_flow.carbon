// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+9]]:11: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+6]]:11: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~~~~~~~
// CHECK:STDERR: fail_control_flow.carbon:[[@LINE+3]]:11: ERROR: Alias initializer must be a template constant.
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
// CHECK:STDOUT:   %.loc16: bool = block_arg <unexpected instblockref block4>
// CHECK:STDOUT:   %a: <error> = bind_template_name a, %.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
