// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_assignment_to_error.carbon:[[@LINE+4]]:3: ERROR: Name `undeclared` not found.
  // CHECK:STDERR:   undeclared = 42;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  undeclared = 42;
  // CHECK:STDERR: fail_assignment_to_error.carbon:[[@LINE+3]]:4: ERROR: Name `also_undeclared` not found.
  // CHECK:STDERR:   *also_undeclared = 42;
  // CHECK:STDERR:    ^~~~~~~~~~~~~~~
  *also_undeclared = 42;
}

// CHECK:STDOUT: --- fail_assignment_to_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Main = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 42 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc12: i32 = int_literal 42 [template = constants.%.2]
// CHECK:STDOUT:   assign %undeclared.ref, <error>
// CHECK:STDOUT:   %also_undeclared.ref: <error> = name_ref also_undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc16_3: ref <error> = deref <error>
// CHECK:STDOUT:   %.loc16_22: i32 = int_literal 42 [template = constants.%.2]
// CHECK:STDOUT:   assign %.loc16_3, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
