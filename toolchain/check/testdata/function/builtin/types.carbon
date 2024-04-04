// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- types.carbon

library "types" api;

fn Int() -> type = "int.make_32";
fn Float(size: i32) -> type = "float.make";
fn Bool() -> type = "bool.make";

// --- use_types.carbon

library "uses_types" api;

import library "types";

var i: Int() = 0;
var f: Float(64) = 0.0;
var b: Bool() = false;

var t: type = Int();

fn GetFloat(dyn_size: i32) -> type {
  return Float(dyn_size);
}

// --- fail_invalid_size.carbon

library "invalid_size" api;

import library "types";

// CHECK:STDERR: fail_invalid_size.carbon:[[@LINE+4]]:20: ERROR: Byte width must be 64.
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
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %Int
// CHECK:STDOUT:     .Float = %Float
// CHECK:STDOUT:     .Bool = %Bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int: <function> = fn_decl @Int [template] {
// CHECK:STDOUT:     %return.var.loc4: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Float: <function> = fn_decl @Float [template] {
// CHECK:STDOUT:     %size.loc5_10.1: i32 = param size
// CHECK:STDOUT:     @Float.%size: i32 = bind_name size, %size.loc5_10.1
// CHECK:STDOUT:     %return.var.loc5: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Bool: <function> = fn_decl @Bool [template] {
// CHECK:STDOUT:     %return.var.loc6: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int() -> type = "int.make_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %.3: f64 = real_literal 0e-1 [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     .Bool = %import_ref.3
// CHECK:STDOUT:     .i = %i
// CHECK:STDOUT:     .f = %f
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .t = %t
// CHECK:STDOUT:     .GetFloat = %GetFloat
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir2, inst+2, loc_11 [template = imports.%Int]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir2, inst+6, loc_20 [template = imports.%Float]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir2, inst+8, loc_30 [template = imports.%Bool]
// CHECK:STDOUT:   %Int.ref: <function> = name_ref Int, %import_ref.1 [template = imports.%Int]
// CHECK:STDOUT:   %.loc6_11.1: init type = call %Int.ref() [template = i32]
// CHECK:STDOUT:   %.loc6_12: type = value_of_initializer %.loc6_11.1 [template = i32]
// CHECK:STDOUT:   %.loc6_11.2: type = converted %.loc6_11.1, %.loc6_12 [template = i32]
// CHECK:STDOUT:   %i.var: ref i32 = var i
// CHECK:STDOUT:   %i: ref i32 = bind_name i, %i.var
// CHECK:STDOUT:   %Float.ref: <function> = name_ref Float, %import_ref.2 [template = imports.%Float]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_13.1: init type = call %Float.ref(%.loc7_14) [template = f64]
// CHECK:STDOUT:   %.loc7_16: type = value_of_initializer %.loc7_13.1 [template = f64]
// CHECK:STDOUT:   %.loc7_13.2: type = converted %.loc7_13.1, %.loc7_16 [template = f64]
// CHECK:STDOUT:   %f.var: ref f64 = var f
// CHECK:STDOUT:   %f: ref f64 = bind_name f, %f.var
// CHECK:STDOUT:   %Bool.ref: <function> = name_ref Bool, %import_ref.3 [template = imports.%Bool]
// CHECK:STDOUT:   %.loc8_12.1: init type = call %Bool.ref() [template = bool]
// CHECK:STDOUT:   %.loc8_13: type = value_of_initializer %.loc8_12.1 [template = bool]
// CHECK:STDOUT:   %.loc8_12.2: type = converted %.loc8_12.1, %.loc8_13 [template = bool]
// CHECK:STDOUT:   %b.var: ref bool = var b
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT:   %t.var: ref type = var t
// CHECK:STDOUT:   %t: ref type = bind_name t, %t.var
// CHECK:STDOUT:   %GetFloat: <function> = fn_decl @GetFloat [template] {
// CHECK:STDOUT:     %dyn_size.loc12_13.1: i32 = param dyn_size
// CHECK:STDOUT:     @GetFloat.%dyn_size: i32 = bind_name dyn_size, %dyn_size.loc12_13.1
// CHECK:STDOUT:     %return.var: ref type = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int() -> type = "int.make_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GetFloat(%dyn_size: i32) -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Float.ref: <function> = name_ref Float, file.%import_ref.2 [template = imports.%Float]
// CHECK:STDOUT:   %dyn_size.ref: i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %.loc13_15.1: init type = call %Float.ref(%dyn_size.ref)
// CHECK:STDOUT:   %.loc13_25: type = value_of_initializer %.loc13_15.1
// CHECK:STDOUT:   %.loc13_15.2: type = converted %.loc13_15.1, %.loc13_25
// CHECK:STDOUT:   return %.loc13_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%i.var, %.loc6
// CHECK:STDOUT:   %.loc7: f64 = real_literal 0e-1 [template = constants.%.3]
// CHECK:STDOUT:   assign file.%f.var, %.loc7
// CHECK:STDOUT:   %.loc8: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   assign file.%b.var, %.loc8
// CHECK:STDOUT:   %Int.ref: <function> = name_ref Int, file.%import_ref.1 [template = imports.%Int]
// CHECK:STDOUT:   %.loc10: init type = call %Int.ref() [template = i32]
// CHECK:STDOUT:   assign file.%t.var, %.loc10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_invalid_size.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 32 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 64 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     .Bool = %import_ref.3
// CHECK:STDOUT:     .invalid_float = %invalid_float
// CHECK:STDOUT:     .dyn_size = %dyn_size
// CHECK:STDOUT:     .dyn = %dyn
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir2, inst+6, loc_11 [template = imports.%Float]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir2, inst+8, unloaded
// CHECK:STDOUT:   %Float.ref.loc10: <function> = name_ref Float, %import_ref.2 [template = imports.%Float]
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 32 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_25.1: init type = call %Float.ref.loc10(%.loc10_26) [template = <error>]
// CHECK:STDOUT:   %.loc10_28: type = value_of_initializer %.loc10_25.1 [template = <error>]
// CHECK:STDOUT:   %.loc10_25.2: type = converted %.loc10_25.1, %.loc10_28 [template = <error>]
// CHECK:STDOUT:   %invalid_float.var: ref <error> = var invalid_float
// CHECK:STDOUT:   %invalid_float: ref <error> = bind_name invalid_float, %invalid_float.var
// CHECK:STDOUT:   %dyn_size.var: ref i32 = var dyn_size
// CHECK:STDOUT:   %dyn_size: ref i32 = bind_name dyn_size, %dyn_size.var
// CHECK:STDOUT:   %Float.ref.loc16: <function> = name_ref Float, %import_ref.2 [template = imports.%Float]
// CHECK:STDOUT:   %dyn_size.ref: ref i32 = name_ref dyn_size, %dyn_size
// CHECK:STDOUT:   %.loc16_16: i32 = bind_value %dyn_size.ref
// CHECK:STDOUT:   %.loc16_15.1: init type = call %Float.ref.loc16(%.loc16_16)
// CHECK:STDOUT:   %.loc16_24: type = value_of_initializer %.loc16_15.1
// CHECK:STDOUT:   %.loc16_15.2: type = converted %.loc16_15.1, %.loc16_24
// CHECK:STDOUT:   %dyn.var: ref <error> = var dyn
// CHECK:STDOUT:   %dyn: ref <error> = bind_name dyn, %dyn.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 64 [template = constants.%.2]
// CHECK:STDOUT:   assign file.%dyn_size.var, %.loc12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
