// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit;

var a_orig: i32 = 0;
var a_ref: i32* = &a_orig;

// --- implicit.impl.carbon

impl package Implicit;

var a: i32* = a_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a_orig = %a_orig
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc4: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc4_13.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:   %.loc4_13.2: type = converted %int.make_type_32.loc4, %.loc4_13.1 [template = i32]
// CHECK:STDOUT:   %a_orig.var: ref i32 = var a_orig
// CHECK:STDOUT:   %a_orig: ref i32 = bind_name a_orig, %a_orig.var
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32.loc5: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc5_15.1: type = value_of_initializer %int.make_type_32.loc5 [template = i32]
// CHECK:STDOUT:   %.loc5_15.2: type = converted %int.make_type_32.loc5, %.loc5_15.1 [template = i32]
// CHECK:STDOUT:   %.loc5_15.3: type = ptr_type i32 [template = constants.%.3]
// CHECK:STDOUT:   %a_ref.var: ref i32* = var a_ref
// CHECK:STDOUT:   %a_ref: ref i32* = bind_name a_ref, %a_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%a_orig.var, %.loc4
// CHECK:STDOUT:   %a_orig.ref: ref i32 = name_ref a_orig, file.%a_orig
// CHECK:STDOUT:   %.loc5: i32* = addr_of %a_orig.ref
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_orig = %import_ref.1
// CHECK:STDOUT:     .a_ref = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir0, inst+12, unloaded
// CHECK:STDOUT:   %import_ref.2: ref i32* = import_ref ir0, inst+23, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_32, %.loc4_11.1 [template = i32]
// CHECK:STDOUT:   %.loc4_11.3: type = ptr_type i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref i32* = var a
// CHECK:STDOUT:   %a: ref i32* = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref i32* = name_ref a_ref, file.%import_ref.2
// CHECK:STDOUT:   %.loc4: i32* = bind_value %a_ref.ref
// CHECK:STDOUT:   assign file.%a.var, %.loc4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
