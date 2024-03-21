// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

let a:! bool = true;

// --- implicit.impl.carbon

package Implicit impl;

let b:! bool = a;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc4: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   %a: bool = bind_symbolic_name a, %.loc4 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: bool = import_ref ir1, inst+1, used [symbolic = <unexpected instref inst+3>]
// CHECK:STDOUT:   %a.ref: bool = name_ref a, %import_ref [symbolic = <unexpected instref inst+3>]
// CHECK:STDOUT:   %b: bool = bind_symbolic_name b, %a.ref [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
