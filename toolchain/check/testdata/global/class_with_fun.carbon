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
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .ret_a = %ret_a
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %ret_a: <function> = fn_decl @ret_a [template] {
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
// CHECK:STDOUT:   %.loc9_11.2: init A = class_init (), %return [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_11.3: init A = converted %.loc9_11.1, %.loc9_11.2 [template = constants.%.4]
// CHECK:STDOUT:   return %.loc9_11.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_13.1: {} = struct_literal ()
// CHECK:STDOUT:   %.loc12_13.2: init A = class_init (), file.%a.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_13.3: init A = converted %.loc12_13.1, %.loc12_13.2 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%a.var, %.loc12_13.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
