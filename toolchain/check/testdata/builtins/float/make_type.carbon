// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- types.carbon

library "types" api;

fn Float(size: i32) -> type = "float.make_type";

// --- use_types.carbon

library "uses_types" api;

import library "types";

var f: Float(64) = 0.0;

fn GetFloat(dyn_size: i32) -> type {
  return Float(dyn_size);
}

// --- fail_invalid_size.carbon

library "invalid_size" api;

import library "types";

// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+4]]:20: ERROR: Bit width must be 64.
// CHECK:STDERR: var invalid_float: Float(32);
// CHECK:STDERR:                    ^~~~~~
// CHECK:STDERR:
var invalid_float: Float(32);

var dyn_size: i32 = 64;
// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+3]]:10: ERROR: Cannot evaluate type expression.
// CHECK:STDERR: var dyn: Float(dyn_size);
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
var dyn: Float(dyn_size);

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Float = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Float = %Float.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Float.decl: Float = fn_decl @Float [template = constants.%struct] {
// CHECK:STDOUT:     %size.loc4_10.1: i32 = param size
// CHECK:STDOUT:     @Float.%size: i32 = bind_name size, %size.loc4_10.1
// CHECK:STDOUT:     @Float.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Float = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %GetFloat: type = fn_type @GetFloat [template]
// CHECK:STDOUT:   %struct.2: GetFloat = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Float = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .GetFloat = %GetFloat.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: Float = import_ref ir1, inst+5, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Float.ref: Float = name_ref Float, %import_ref [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc6_14: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%.loc6_14) [template = f64]
// CHECK:STDOUT:   %.loc6_16.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc6_16.2: type = converted %float.make_type, %.loc6_16.1 [template = f64]
// CHECK:STDOUT:   %f.var: ref f64 = var f
// CHECK:STDOUT:   %f: ref f64 = bind_name f, %f.var
// CHECK:STDOUT:   %GetFloat.decl: GetFloat = fn_decl @GetFloat [template = constants.%struct.2] {
// CHECK:STDOUT:     %dyn_size.loc8_13.1: i32 = param dyn_size
// CHECK:STDOUT:     @GetFloat.%dyn_size: i32 = bind_name dyn_size, %dyn_size.loc8_13.1
// CHECK:STDOUT:     @GetFloat.%return: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetFloat(%dyn_size: i32) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float.ref: Float = name_ref Float, file.%import_ref [template = constants.%struct.1]
// CHECK:STDOUT:   %dyn_size.ref: i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %float.make_type: init type = call %Float.ref(%dyn_size.ref)
// CHECK:STDOUT:   %.loc9_25.1: type = value_of_initializer %float.make_type
// CHECK:STDOUT:   %.loc9_25.2: type = converted %float.make_type, %.loc9_25.1
// CHECK:STDOUT:   return %.loc9_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: f64 = float_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   assign file.%f.var, %.loc6
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Float: type = fn_type @Float [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Float = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 64 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Float = %import_ref
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .invalid_float = %invalid_float
// CHECK:STDOUT:     .dyn_size = %dyn_size
// CHECK:STDOUT:     .dyn = %dyn
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: Float = import_ref ir1, inst+5, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Float.ref.loc10: Float = name_ref Float, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 32 [template = constants.%.2]
// CHECK:STDOUT:   %float.make_type.loc10: init type = call %Float.ref.loc10(%.loc10_26) [template = <error>]
// CHECK:STDOUT:   %.loc10_28.1: type = value_of_initializer %float.make_type.loc10 [template = <error>]
// CHECK:STDOUT:   %.loc10_28.2: type = converted %float.make_type.loc10, %.loc10_28.1 [template = <error>]
// CHECK:STDOUT:   %invalid_float.var: ref <error> = var invalid_float
// CHECK:STDOUT:   %invalid_float: ref <error> = bind_name invalid_float, %invalid_float.var
// CHECK:STDOUT:   %dyn_size.var: ref i32 = var dyn_size
// CHECK:STDOUT:   %dyn_size: ref i32 = bind_name dyn_size, %dyn_size.var
// CHECK:STDOUT:   %Float.ref.loc16: Float = name_ref Float, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %dyn_size.ref: ref i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %.loc16_16: i32 = bind_value %dyn_size.ref
// CHECK:STDOUT:   %float.make_type.loc16: init type = call %Float.ref.loc16(%.loc16_16)
// CHECK:STDOUT:   %.loc16_24.1: type = value_of_initializer %float.make_type.loc16
// CHECK:STDOUT:   %.loc16_24.2: type = converted %float.make_type.loc16, %.loc16_24.1
// CHECK:STDOUT:   %dyn.var: ref <error> = var dyn
// CHECK:STDOUT:   %dyn: ref <error> = bind_name dyn, %dyn.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 64 [template = constants.%.3]
// CHECK:STDOUT:   assign file.%dyn_size.var, %.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
