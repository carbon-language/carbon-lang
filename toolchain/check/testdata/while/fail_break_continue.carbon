// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn While() {
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: ERROR: `continue` can only be used in a loop.
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: ERROR: `break` can only be used in a loop.
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  break;
  while (false) {
    continue;
    break;
  }
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: ERROR: `continue` can only be used in a loop.
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+3]]:3: ERROR: `break` can only be used in a loop.
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  break;
}

// CHECK:STDOUT: --- fail_break_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %While: type = fn_type @While [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: While = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %While.decl: While = fn_decl @While [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
