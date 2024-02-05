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

fn F() -> const i32;

var a_ref: const i32 = F();

// --- implicit.impl.carbon

package Implicit impl;

var a: const i32 = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = const_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .a_ref = %a_ref} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %.loc6: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:   %a_ref.var: ref const i32 = var a_ref
// CHECK:STDOUT:   %a_ref: ref const i32 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> const i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc6: init const i32 = call %F.ref()
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = const_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %import_ref.1, .a_ref = %import_ref.2, .a = %a} [template]
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+4, unused
// CHECK:STDOUT:   %import_ref.2: ref <error> = import_ref ir1, inst+7, used
// CHECK:STDOUT:   %.loc4: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref const i32 = var a
// CHECK:STDOUT:   %a: ref const i32 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref <error> = name_ref a_ref, file.%import_ref.2
// CHECK:STDOUT:   assign file.%a.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
