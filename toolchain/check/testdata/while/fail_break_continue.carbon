// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn While() {
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+3]]:3: ERROR: `continue` can only be used in a loop.
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+3]]:3: ERROR: `break` can only be used in a loop.
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  break;
  while (false) {
    continue;
    break;
  }
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+3]]:3: ERROR: `continue` can only be used in a loop.
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+3]]:3: ERROR: `break` can only be used in a loop.
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  break;
}

// CHECK:STDOUT: --- fail_break_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.While = %While} [template]
// CHECK:STDOUT:   %While: <function> = fn_decl @While [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
