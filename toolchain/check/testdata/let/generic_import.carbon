// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

let T:! type = i32;

// --- implicit.impl.carbon

package Implicit impl;

var a: T*;
var b: T = *a;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, i32 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .T = %import_ref
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: type = import_ref ir1, inst+1, used [symbolic = <unexpected instref inst+2>]
// CHECK:STDOUT:   %T.ref.loc4: type = name_ref T, %import_ref [symbolic = <unexpected instref inst+2>]
// CHECK:STDOUT:   %.loc4: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:   %a.var: ref T* = var a
// CHECK:STDOUT:   %a: ref T* = bind_name a, %a.var
// CHECK:STDOUT:   %T.ref.loc5: type = name_ref T, %import_ref [symbolic = <unexpected instref inst+2>]
// CHECK:STDOUT:   %b.var: ref T = var b
// CHECK:STDOUT:   %b: ref T = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: ref T* = name_ref a, file.%a
// CHECK:STDOUT:   %.loc5_13: T* = bind_value %a.ref
// CHECK:STDOUT:   %.loc5_12.1: ref T = deref %.loc5_13
// CHECK:STDOUT:   %.loc5_12.2: T = bind_value %.loc5_12.1
// CHECK:STDOUT:   assign file.%b.var, %.loc5_12.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
