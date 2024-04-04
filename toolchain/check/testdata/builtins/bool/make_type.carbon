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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Bool = %Bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bool: <function> = fn_decl @Bool [template] {
// CHECK:STDOUT:     %return.var: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <function> = import_ref ir2, inst+2, loc_11 [template = imports.%Bool]
// CHECK:STDOUT:   %Bool.ref: <function> = name_ref Bool, %import_ref [template = imports.%Bool]
// CHECK:STDOUT:   %.loc6_12.1: init type = call %Bool.ref() [template = bool]
// CHECK:STDOUT:   %.loc6_13: type = value_of_initializer %.loc6_12.1 [template = bool]
// CHECK:STDOUT:   %.loc6_12.2: type = converted %.loc6_12.1, %.loc6_13 [template = bool]
// CHECK:STDOUT:   %b.var: ref bool = var b
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   assign file.%b.var, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
