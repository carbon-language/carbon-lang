// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn MakeI32() -> i32;
fn MakeF64() -> f64;

fn F() -> {.a: i32, .b: f64} {
  let x: {.a: i32, .b: f64} = {.b = MakeF64(), .a = MakeI32()};
  let y: {.b: f64, .a: i32} = x;
  return y;
}

// CHECK:STDOUT: --- reorder_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32, .b: f64} [template]
// CHECK:STDOUT:   %.2: type = ptr_type {.a: i32, .b: f64} [template]
// CHECK:STDOUT:   %.3: type = struct_type {.b: f64, .a: i32} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.b: f64, .a: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .MakeI32 = %MakeI32
// CHECK:STDOUT:     .MakeF64 = %MakeF64
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %MakeI32: <function> = fn_decl @MakeI32 [template] {
// CHECK:STDOUT:     @MakeI32.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeF64: <function> = fn_decl @MakeF64 [template] {
// CHECK:STDOUT:     @MakeF64.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %.loc10: type = struct_type {.a: i32, .b: f64} [template = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref {.a: i32, .b: f64} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeI32() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeF64() -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: {.a: i32, .b: f64} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_27: type = struct_type {.a: i32, .b: f64} [template = constants.%.1]
// CHECK:STDOUT:   %MakeF64.ref: <function> = name_ref MakeF64, file.%MakeF64 [template = file.%MakeF64]
// CHECK:STDOUT:   %MakeF64.call: init f64 = call %MakeF64.ref()
// CHECK:STDOUT:   %MakeI32.ref: <function> = name_ref MakeI32, file.%MakeI32 [template = file.%MakeI32]
// CHECK:STDOUT:   %MakeI32.call: init i32 = call %MakeI32.ref()
// CHECK:STDOUT:   %.loc11_62.1: {.b: f64, .a: i32} = struct_literal (%MakeF64.call, %MakeI32.call)
// CHECK:STDOUT:   %.loc11_62.2: i32 = value_of_initializer %MakeI32.call
// CHECK:STDOUT:   %.loc11_62.3: i32 = converted %MakeI32.call, %.loc11_62.2
// CHECK:STDOUT:   %.loc11_62.4: f64 = value_of_initializer %MakeF64.call
// CHECK:STDOUT:   %.loc11_62.5: f64 = converted %MakeF64.call, %.loc11_62.4
// CHECK:STDOUT:   %.loc11_62.6: {.a: i32, .b: f64} = struct_value (%.loc11_62.3, %.loc11_62.5)
// CHECK:STDOUT:   %.loc11_63: {.a: i32, .b: f64} = converted %.loc11_62.1, %.loc11_62.6
// CHECK:STDOUT:   %x: {.a: i32, .b: f64} = bind_name x, %.loc11_63
// CHECK:STDOUT:   %.loc12_27: type = struct_type {.b: f64, .a: i32} [template = constants.%.3]
// CHECK:STDOUT:   %x.ref: {.a: i32, .b: f64} = name_ref x, %x
// CHECK:STDOUT:   %.loc12_31.1: f64 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc12_31.2: i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc12_31.3: {.b: f64, .a: i32} = struct_value (%.loc12_31.1, %.loc12_31.2)
// CHECK:STDOUT:   %.loc12_32: {.b: f64, .a: i32} = converted %x.ref, %.loc12_31.3
// CHECK:STDOUT:   %y: {.b: f64, .a: i32} = bind_name y, %.loc12_32
// CHECK:STDOUT:   %y.ref: {.b: f64, .a: i32} = name_ref y, %y
// CHECK:STDOUT:   %.loc13_10.1: i32 = struct_access %y.ref, element1
// CHECK:STDOUT:   %.loc13_10.2: ref i32 = struct_access %return, element1
// CHECK:STDOUT:   %.loc13_10.3: init i32 = initialize_from %.loc13_10.1 to %.loc13_10.2
// CHECK:STDOUT:   %.loc13_10.4: f64 = struct_access %y.ref, element0
// CHECK:STDOUT:   %.loc13_10.5: ref f64 = struct_access %return, element0
// CHECK:STDOUT:   %.loc13_10.6: init f64 = initialize_from %.loc13_10.4 to %.loc13_10.5
// CHECK:STDOUT:   %.loc13_10.7: init {.a: i32, .b: f64} = struct_init (%.loc13_10.3, %.loc13_10.6) to %return
// CHECK:STDOUT:   %.loc13_11: init {.a: i32, .b: f64} = converted %y.ref, %.loc13_10.7
// CHECK:STDOUT:   return %.loc13_11
// CHECK:STDOUT: }
// CHECK:STDOUT:
