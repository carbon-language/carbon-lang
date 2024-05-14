// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
class A {}

fn ret_a() -> A {
  return {};
}

var a: A = {};

// CHECK:STDOUT: --- class_with_fun.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %ret_a: type = fn_type @ret_a [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: ret_a = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %struct.2: A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .ret_a = %ret_a.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %ret_a.decl: ret_a = fn_decl @ret_a [template = constants.%struct.1] {
// CHECK:STDOUT:     %A.ref.loc8: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     @ret_a.%return: ref A = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref.loc12: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %a.var: ref A = var a
// CHECK:STDOUT:   %a: ref A = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ret_a() -> %return: A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_11.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc9_11.2: init A = class_init (), %return [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_12: init A = converted %.loc9_11.1, %.loc9_11.2 [template = constants.%struct.2]
// CHECK:STDOUT:   return %.loc9_12 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc12_13.2: init A = class_init (), file.%a.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc12_14: init A = converted %.loc12_13.1, %.loc12_13.2 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%a.var, %.loc12_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
