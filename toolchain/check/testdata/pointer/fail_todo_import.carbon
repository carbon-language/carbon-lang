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

var a_orig: i32 = 0;
var a_ref: i32* = &a_orig;

// --- implicit.impl.carbon

package Implicit impl;

var a: i32* = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_orig = %a_orig, .a_ref = %a_ref} [template]
// CHECK:STDOUT:   %a_orig.var: ref i32 = var a_orig
// CHECK:STDOUT:   %a_orig: ref i32 = bind_name a_orig, %a_orig.var
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign %a_orig.var, %.loc4
// CHECK:STDOUT:   %.loc5_15: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref i32* = var a_ref
// CHECK:STDOUT:   %a_ref: ref i32* = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %a_orig.ref: ref i32 = name_ref a_orig, %a_orig
// CHECK:STDOUT:   %.loc5_19: i32* = addr_of %a_orig.ref
// CHECK:STDOUT:   assign %a_ref.var, %.loc5_19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a_orig = %import_ref.1, .a_ref = %import_ref.2, .a = %a} [template]
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unused
// CHECK:STDOUT:   %import_ref.2: ref <error> = import_ref ir1, inst+9, used
// CHECK:STDOUT:   %.loc4: type = ptr_type i32 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref i32* = var a
// CHECK:STDOUT:   %a: ref i32* = bind_name a, %a.var
// CHECK:STDOUT:   %a_ref.ref: ref <error> = name_ref a_ref, %import_ref.2
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
