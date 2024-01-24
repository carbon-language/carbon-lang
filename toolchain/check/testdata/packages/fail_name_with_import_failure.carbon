// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.impl.carbon

// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:1: ERROR: Corresponding API not found.
// CHECK:STDERR: package Implicit impl;
// CHECK:STDERR: ^~~~~~~
package Implicit impl;

var a: () = A();

// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, has_error} [template]
// CHECK:STDOUT:   %.loc7_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A.ref: <error> = name_ref A, <error> [template = <error>]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
