// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: Should this be valid?
fn F(a: i32) -> i32 {
  let T:! type = i32;
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+4]]:3: ERROR: Cannot implicitly convert from `i32` to `T`.
  // CHECK:STDERR:   let x: T = 5;
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let x: T = 5;
  // CHECK:STDERR: fail_generic.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `T` to `i32`.
  // CHECK:STDERR:   return x;
  // CHECK:STDERR:   ^~~~~~~~~
  return x;
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.2: i32 = int_literal 5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %a.loc8_6.1: i32 = param a
// CHECK:STDOUT:     @F.%a: i32 = bind_name a, %a.loc8_6.1
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0, i32 [symbolic = constants.%T]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc14: i32 = int_literal 5 [template = constants.%.2]
// CHECK:STDOUT:   %x: T = bind_name x, <error>
// CHECK:STDOUT:   %x.ref: T = name_ref x, %x
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
