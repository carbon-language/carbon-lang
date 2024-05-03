// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: i32) -> i32 {
  let a: i32 = 1;
  // TODO: Reject this due to ambiguity.
  return a;
}

// CHECK:STDOUT: --- shadowed_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %a.loc7_6.1: i32 = param a
// CHECK:STDOUT:     @F.%a.loc7: i32 = bind_name a, %a.loc7_6.1
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.loc7: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %a.loc8: i32 = bind_name a, %.loc8
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a.loc8
// CHECK:STDOUT:   return %a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
