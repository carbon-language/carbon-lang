// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// When there are no more cases that can hit a TODO, remove this test. Until
// then, update it whenever its target is implemented.
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

var a_ref: i32 = 0;

// --- implicit.impl.carbon

package Implicit impl;

var a: i32 = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %a_ref} [template]
// CHECK:STDOUT:   %a_ref.var: ref i32 = var a_ref
// CHECK:STDOUT:   %a_ref: ref i32 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %a_ref.var, %.loc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %import_ref, .a = %a} [template]
// CHECK:STDOUT:   %import_ref: ref i32 = import_ref ir1, inst+2, used
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %a_ref.ref: ref i32 = name_ref a_ref, %import_ref
// CHECK:STDOUT:   %.loc4: i32 = bind_value %a_ref.ref
// CHECK:STDOUT:   assign %a.var, %.loc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
