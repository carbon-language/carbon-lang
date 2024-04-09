// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- types.carbon

library "types" api;

fn Int() -> type = "int.make_type_32";

// --- use_types.carbon

library "uses_types" api;

import library "types";

var i: Int() = 0;

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Int = %Int
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int: <function> = fn_decl @Int [template] {
// CHECK:STDOUT:     @Int.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .i = %i
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir2, inst+3, loc_11 [template = imports.%Int]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Int.ref: <function> = name_ref Int, %import_ref [template = imports.%Int]
// CHECK:STDOUT:   %.loc6_11.1: init type = call %Int.ref() [template = i32]
// CHECK:STDOUT:   %.loc6_12: type = value_of_initializer %.loc6_11.1 [template = i32]
// CHECK:STDOUT:   %.loc6_11.2: type = converted %.loc6_11.1, %.loc6_12 [template = i32]
// CHECK:STDOUT:   %i.var: ref i32 = var i
// CHECK:STDOUT:   %i: ref i32 = bind_name i, %i.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%i.var, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
