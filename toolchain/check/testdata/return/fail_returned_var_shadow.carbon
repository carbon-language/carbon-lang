// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn SameScope() -> i32 {
  if (true) {
    returned var v: i32 = 0;
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE+7]]:18: ERROR: Cannot declare a `returned var` in the scope of another `returned var`.
    // CHECK:STDERR:     returned var w: i32 = 1;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR: fail_returned_var_shadow.carbon:[[@LINE-4]]:18: `returned var` was declared here.
    // CHECK:STDERR:     returned var v: i32 = 0;
    // CHECK:STDERR:                  ^
    // CHECK:STDERR:
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
// CHECK:STDOUT:   %SameScope: type = fn_type @SameScope [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: SameScope = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %DifferentScopes: type = fn_type @DifferentScopes [template]
// CHECK:STDOUT:   %struct.2: DifferentScopes = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .SameScope = %SameScope.decl
// CHECK:STDOUT:     .DifferentScopes = %DifferentScopes.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %SameScope.decl: SameScope = fn_decl @SameScope [template = constants.%struct.1] {
// CHECK:STDOUT:     @SameScope.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentScopes.decl: DifferentScopes = fn_decl @DifferentScopes [template = constants.%struct.2] {
// CHECK:STDOUT:     @DifferentScopes.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SameScope() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc8 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc9: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc9
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc17: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc17
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc19: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DifferentScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc23 br !if.then.loc23 else br !if.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc23:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc24: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, %.loc24
// CHECK:STDOUT:   %.loc25: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc25 br !if.then.loc25 else br !if.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc25:
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc32: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   assign %w.var, %.loc32
// CHECK:STDOUT:   br !if.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc25:
// CHECK:STDOUT:   br !if.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc23:
// CHECK:STDOUT:   %.loc35: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc35
// CHECK:STDOUT: }
// CHECK:STDOUT:
