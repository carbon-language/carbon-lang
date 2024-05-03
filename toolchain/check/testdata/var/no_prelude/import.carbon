// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

var a_ref: () = ();

// --- implicit.impl.carbon

package Implicit impl;

var a: () = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_13.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_13.2: type = converted %.loc4_13.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a_ref.var: ref () = var a_ref
// CHECK:STDOUT:   %a_ref: ref () = bind_name a_ref, %a_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_18.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_18.2: init () = tuple_init () to file.%a_ref.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_19: init () = converted %.loc4_18.1, %.loc4_18.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_19
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = %import_ref
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: ref () = import_ref ir0, inst+5, loaded
// CHECK:STDOUT:   %.loc4_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref () = name_ref a_ref, file.%import_ref
// CHECK:STDOUT:   %.loc4_13: init () = tuple_init () to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc4_18: init () = converted %a_ref.ref, %.loc4_13 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc4_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
