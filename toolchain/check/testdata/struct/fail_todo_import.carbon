// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// When there are no more cases that can hit a TODO, remove this test. Until
// then, update it whenever its target is implemented.
//
// AUTOUPDATE
// CHECK:STDERR: implicit.impl.carbon: ERROR: Semantics TODO: `TODO: ResolveIfImportRefUnused for non-builtin type`.

// --- implicit.carbon

package Implicit api;

var a_ref: {.a: i32} = {.a = 0};

// --- implicit.impl.carbon

package Implicit impl;

var a: {.a: i32} = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: {.a: i32} = struct_value (%.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %a_ref} [template]
// CHECK:STDOUT:   %.loc4_20: type = struct_type {.a: i32} [template = constants.%.1]
// CHECK:STDOUT:   %a_ref.var: ref {.a: i32} = var a_ref
// CHECK:STDOUT:   %a_ref: ref {.a: i32} = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc4_30: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_31.1: {.a: i32} = struct_literal (%.loc4_30)
// CHECK:STDOUT:   %.loc4_31.2: init {.a: i32} = struct_init (%.loc4_30) to %a_ref.var [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_31.3: init {.a: i32} = converted %.loc4_31.1, %.loc4_31.2 [template = constants.%.3]
// CHECK:STDOUT:   assign %a_ref.var, %.loc4_31.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_ref = %import_ref, .a = %a} [template]
// CHECK:STDOUT:   %import_ref: ref <error> = import_ref ir1, inst+6, used
// CHECK:STDOUT:   %.loc4: type = struct_type {.a: i32} [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref {.a: i32} = var a
// CHECK:STDOUT:   %a: ref {.a: i32} = bind_name a, %a.var
// CHECK:STDOUT:   %a_ref.ref: ref <error> = name_ref a_ref, %import_ref
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
