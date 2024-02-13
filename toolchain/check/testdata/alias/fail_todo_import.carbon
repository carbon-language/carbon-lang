// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

var a: i32 = 0;

alias b = a;

// --- b.carbon

// CHECK:STDERR: b.carbon:[[@LINE+3]]:14: ERROR: Name `b` not found.
// CHECK:STDERR: var c: i32 = b;
// CHECK:STDERR:              ^
var c: i32 = b;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b.ref} [template]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: ref i32 = name_ref b, %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%a.var, %.loc4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.c = %c} [template]
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
