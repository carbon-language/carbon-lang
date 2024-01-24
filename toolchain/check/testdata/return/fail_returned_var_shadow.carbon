// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn SameScope() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+6]]:18: ERROR: Cannot declare a `returned var` in the scope of another `returned var`.
    // CHECK:STDERR:     returned var w: i32 = 1;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-4]]:18: `returned var` was declared here.
    // CHECK:STDERR:     returned var v: i32 = 0;
    // CHECK:STDERR:                  ^
    returned var w: i32 = 1;
  }
  return 0;
}

fn DifferentScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    if (true) {
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+6]]:20: ERROR: Cannot declare a `returned var` in the scope of another `returned var`.
      // CHECK:STDERR:       returned var w: i32 = 1;
      // CHECK:STDERR:                    ^
      // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-5]]:18: `returned var` was declared here.
      // CHECK:STDERR:     returned var v: i32 = 0;
      // CHECK:STDERR:                  ^
      returned var w: i32 = 1;
    }
  }
  return 0;
}

// CHECK:STDOUT: --- fail_returned_var_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.SameScope = %SameScope, .DifferentScopes = %DifferentScopes} [template]
// CHECK:STDOUT:   %SameScope: <function> = fn_decl @SameScope [template]
// CHECK:STDOUT:   %DifferentScopes: <function> = fn_decl @DifferentScopes [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SameScope() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc8 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc9: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc9
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc16: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %w.var, %.loc16
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc18: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DifferentScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc22 br !if.then.loc22 else br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc22:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc23: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc23
// CHECK:STDOUT:   %.loc24: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc24 br !if.then.loc24 else br !if.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc24:
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc31: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %w.var, %.loc31
// CHECK:STDOUT:   br !if.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc24:
// CHECK:STDOUT:   br !if.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc22:
// CHECK:STDOUT:   %.loc34: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc34
// CHECK:STDOUT: }
// CHECK:STDOUT:
