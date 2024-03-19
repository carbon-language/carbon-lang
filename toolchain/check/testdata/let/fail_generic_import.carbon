// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

let T:! type = i32;

// --- fail_implicit.impl.carbon

package Implicit impl;

// TODO: Should this be valid?
// CHECK:STDERR: fail_implicit.impl.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `i32` to `T`.
// CHECK:STDERR: let a: T = 0;
// CHECK:STDERR: ^~~~~~~~~~~~~
let a: T = 0;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, i32 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .T = %import_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: type = import_ref ir1, inst+1, used [symbolic = <unexpected instref inst+2>]
// CHECK:STDOUT:   %T.ref: type = name_ref T, %import_ref [symbolic = <unexpected instref inst+2>]
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   %a: T = bind_name a, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
