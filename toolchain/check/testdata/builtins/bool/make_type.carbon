// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- types.carbon

library "types" api;

fn Bool() -> type = "bool.make_type";

// --- use_types.carbon

library "uses_types" api;

import library "types";

var b: Bool() = false;

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Bool = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Bool = %Bool.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Bool.decl: Bool = fn_decl @Bool [template = constants.%struct] {
// CHECK:STDOUT:     @Bool.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Bool = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: Bool = import_ref ir1, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Bool.ref: Bool = name_ref Bool, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %bool.make_type: init type = call %Bool.ref() [template = bool]
// CHECK:STDOUT:   %.loc6_13.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:   %.loc6_13.2: type = converted %bool.make_type, %.loc6_13.1 [template = bool]
// CHECK:STDOUT:   %b.var: ref bool = var b
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   assign file.%b.var, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
