// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// When there are no more cases that can hit a TODO, remove this test. Until
// then, update it whenever its target is implemented.
//
// AUTOUPDATE
// CHECK:STDERR: implicit.impl.carbon: ERROR: Semantics TODO: `TODO: support BindName`.

// --- implicit.carbon

package Implicit api;

var a_ref: i32 = 0;

// --- implicit.impl.carbon

package Implicit impl;

var a: () = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %a_ref}, invalid [template]
// CHECK:STDOUT:   %a_ref.var: ref i32 = var a_ref
// CHECK:STDOUT:   %a_ref: ref i32 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %a_ref.var, %.loc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %package.var, .a = %a}, invalid [template]
// CHECK:STDOUT:   %package.var: ref <error> = var package
// CHECK:STDOUT:   %.loc4_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %a_ref.ref: ref <error> = name_ref a_ref, %package.var
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
