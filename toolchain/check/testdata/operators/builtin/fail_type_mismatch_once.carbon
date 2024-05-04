// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  // The following line has two mismatches, but after the first, it shouldn't
  // keep erroring.
  // CHECK:STDERR: fail_type_mismatch_once.carbon:[[@LINE+15]]:10: ERROR: Name `Add` not found.
  // CHECK:STDERR:   return 12 + 3.4 + 12;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_type_mismatch_once.carbon:[[@LINE+11]]:10: ERROR: Semantics TODO: `missing or invalid operator interface`.
  // CHECK:STDERR:   return 12 + 3.4 + 12;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_type_mismatch_once.carbon:[[@LINE+7]]:10: ERROR: Name `Add` not found.
  // CHECK:STDERR:   return 12 + 3.4 + 12;
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_type_mismatch_once.carbon:[[@LINE+3]]:10: ERROR: Semantics TODO: `missing or invalid operator interface`.
  // CHECK:STDERR:   return 12 + 3.4 + 12;
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  return 12 + 3.4 + 12;
}

// CHECK:STDOUT: --- fail_type_mismatch_once.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Main = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 3.4000000000000004 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct] {
// CHECK:STDOUT:     @Main.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc25_10: i32 = int_literal 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc25_15: f64 = float_literal 3.4000000000000004 [template = constants.%.3]
// CHECK:STDOUT:   %.loc25_21: i32 = int_literal 12 [template = constants.%.2]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
