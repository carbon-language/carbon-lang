// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

fn F() -> const i32;

var a_ref: const i32 = F();
var a_ptr_ref: const i32* = &a_ref;

// --- implicit.impl.carbon

package Implicit impl;

// Take a reference to avoid unsupported copy logic. This still validates the
// `const` is handled.
var a: const i32* = &a_ref;
var a_ptr: const i32* = a_ptr_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = const_type i32 [template]
// CHECK:STDOUT:   %.2: type = ptr_type const i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .a_ptr_ref = %a_ptr_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %.loc4: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %return.var: ref const i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:   %a_ref.var: ref const i32 = var a_ref
// CHECK:STDOUT:   %a_ref: ref const i32 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc7_16: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_25: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:   %a_ptr_ref.var: ref const i32* = var a_ptr_ref
// CHECK:STDOUT:   %a_ptr_ref: ref const i32* = bind_name a_ptr_ref, %a_ptr_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> const i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc6: init const i32 = call %F.ref()
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc6
// CHECK:STDOUT:   %a_ref.ref: ref const i32 = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %.loc7: const i32* = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr_ref.var, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = const_type i32 [template]
// CHECK:STDOUT:   %.2: type = ptr_type const i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %import_ref.1
// CHECK:STDOUT:     .a_ref = %import_ref.2
// CHECK:STDOUT:     .a_ptr_ref = %import_ref.3
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .a_ptr = %a_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+4, unused
// CHECK:STDOUT:   %import_ref.2: ref const i32 = import_ref ir1, inst+7, used
// CHECK:STDOUT:   %import_ref.3: ref const i32* = import_ref ir1, inst+15, used
// CHECK:STDOUT:   %.loc6_8: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc6_17: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref const i32* = var a
// CHECK:STDOUT:   %a: ref const i32* = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_12: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_21: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:   %a_ptr.var: ref const i32* = var a_ptr
// CHECK:STDOUT:   %a_ptr: ref const i32* = bind_name a_ptr, %a_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref const i32 = name_ref a_ref, file.%import_ref.2
// CHECK:STDOUT:   %.loc6: const i32* = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a.var, %.loc6
// CHECK:STDOUT:   %a_ptr_ref.ref: ref const i32* = name_ref a_ptr_ref, file.%import_ref.3
// CHECK:STDOUT:   %.loc7: const i32* = bind_value %a_ptr_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr.var, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
