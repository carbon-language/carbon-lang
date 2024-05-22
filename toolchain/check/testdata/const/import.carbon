// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit;

fn F() -> const i32;

var a_ref: const i32 = F();
var a_ptr_ref: const i32* = &a_ref;

// --- implicit.impl.carbon

impl package Implicit;

// Take a reference to avoid unsupported copy logic. This still validates the
// `const` is handled.
var a: const i32* = &a_ref;
var a_ptr: const i32* = a_ptr_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type const i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .a_ptr_ref = %a_ptr_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_32.loc4, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:     %.loc4_11.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     @F.%return: ref const i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc6_12.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_12.2: type = converted %int.make_type_32.loc6, %.loc6_12.1 [template = i32]
// CHECK:STDOUT:   %.loc6_12.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref const i32 = var a_ref
// CHECK:STDOUT:   %a_ref: ref const i32 = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc7_16.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_16.2: type = converted %int.make_type_32.loc7, %.loc7_16.1 [template = i32]
// CHECK:STDOUT:   %.loc7_16.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_25: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:   %a_ptr_ref.var: ref const i32* = var a_ptr_ref
// CHECK:STDOUT:   %a_ptr_ref: ref const i32* = bind_name a_ptr_ref, %a_ptr_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> const i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %F.call: init const i32 = call %F.ref()
// CHECK:STDOUT:   assign file.%a_ref.var, %F.call
// CHECK:STDOUT:   %a_ref.ref: ref const i32 = name_ref a_ref, file.%a_ref
// CHECK:STDOUT:   %.loc7: const i32* = addr_of %a_ref.ref
// CHECK:STDOUT:   assign file.%a_ptr_ref.var, %.loc7
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type const i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %import_ref.1
// CHECK:STDOUT:     .a_ref = %import_ref.2
// CHECK:STDOUT:     .a_ptr_ref = %import_ref.3
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .a_ptr = %a_ptr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir0, inst+14, unloaded
// CHECK:STDOUT:   %import_ref.2: ref const i32 = import_ref ir0, inst+23, loaded
// CHECK:STDOUT:   %import_ref.3: ref const i32* = import_ref ir0, inst+35, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc6: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_32.loc6, %.loc6_8.1 [template = i32]
// CHECK:STDOUT:   %.loc6_8.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %.loc6_17: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref const i32* = var a
// CHECK:STDOUT:   %a: ref const i32* = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc7_12.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_12.2: type = converted %int.make_type_32.loc7, %.loc7_12.1 [template = i32]
// CHECK:STDOUT:   %.loc7_12.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_21: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:   %a_ptr.var: ref const i32* = var a_ptr
// CHECK:STDOUT:   %a_ptr: ref const i32* = bind_name a_ptr, %a_ptr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
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
