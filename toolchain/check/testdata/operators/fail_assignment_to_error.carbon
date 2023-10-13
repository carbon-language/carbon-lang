// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_assignment_to_error.carbon:[[@LINE+3]]:3: ERROR: Name `undeclared` not found.
  // CHECK:STDERR:   undeclared = 42;
  // CHECK:STDERR:   ^
  undeclared = 42;
  // CHECK:STDERR: fail_assignment_to_error.carbon:[[@LINE+3]]:4: ERROR: Name `also_undeclared` not found.
  // CHECK:STDERR:   *also_undeclared = 42;
  // CHECK:STDERR:    ^
  *also_undeclared = 42;
}

// CHECK:STDOUT: file "fail_assignment_to_error.carbon" {
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref: <error> = name_reference "undeclared", <error>
// CHECK:STDOUT:   %.loc11: i32 = int_literal 42
// CHECK:STDOUT:   assign %undeclared.ref, <error>
// CHECK:STDOUT:   %also_undeclared.ref: <error> = name_reference "also_undeclared", <error>
// CHECK:STDOUT:   %.loc15_3: ref <error> = dereference <error>
// CHECK:STDOUT:   %.loc15_22: i32 = int_literal 42
// CHECK:STDOUT:   assign %.loc15_3, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
