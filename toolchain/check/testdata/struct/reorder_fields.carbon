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

// CHECK:STDOUT: file "reorder_fields.carbon" {
// CHECK:STDOUT:   %MakeI32: <function> = fn_decl @MakeI32
// CHECK:STDOUT:   %MakeF64: <function> = fn_decl @MakeF64
// CHECK:STDOUT:   %.loc10: type = ptr_type {.a: i32, .b: f64}
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeI32() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeF64() -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: {.a: i32, .b: f64} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_27: type = struct_type {.a: i32, .b: f64}
// CHECK:STDOUT:   %MakeF64.ref: <function> = name_reference "MakeF64", file.%MakeF64
// CHECK:STDOUT:   %.loc11_44: init f64 = call %MakeF64.ref()
// CHECK:STDOUT:   %MakeI32.ref: <function> = name_reference "MakeI32", file.%MakeI32
// CHECK:STDOUT:   %.loc11_60: init i32 = call %MakeI32.ref()
// CHECK:STDOUT:   %.loc11_62.1: type = struct_type {.b: f64, .a: i32}
// CHECK:STDOUT:   %.loc11_62.2: {.b: f64, .a: i32} = struct_literal (%.loc11_44, %.loc11_60)
// CHECK:STDOUT:   %.loc11_62.3: i32 = value_of_initializer %.loc11_60
// CHECK:STDOUT:   %.loc11_62.4: f64 = value_of_initializer %.loc11_44
// CHECK:STDOUT:   %.loc11_62.5: {.a: i32, .b: f64} = struct_value %.loc11_62.2, (%.loc11_62.3, %.loc11_62.4)
// CHECK:STDOUT:   %x: {.a: i32, .b: f64} = bind_name "x", %.loc11_62.5
// CHECK:STDOUT:   %.loc12_27: type = struct_type {.b: f64, .a: i32}
// CHECK:STDOUT:   %.loc11_62.6: type = ptr_type {.b: f64, .a: i32}
// CHECK:STDOUT:   %x.ref: {.a: i32, .b: f64} = name_reference "x", %x
// CHECK:STDOUT:   %.loc12_31.1: f64 = struct_access %x.ref, member1
// CHECK:STDOUT:   %.loc12_31.2: i32 = struct_access %x.ref, member0
// CHECK:STDOUT:   %.loc12_31.3: {.b: f64, .a: i32} = struct_value %x.ref, (%.loc12_31.1, %.loc12_31.2)
// CHECK:STDOUT:   %y: {.b: f64, .a: i32} = bind_name "y", %.loc12_31.3
// CHECK:STDOUT:   %y.ref: {.b: f64, .a: i32} = name_reference "y", %y
// CHECK:STDOUT:   %.loc13_10.1: i32 = struct_access %y.ref, member1
// CHECK:STDOUT:   %.loc13_10.2: ref i32 = struct_access %return, member1
// CHECK:STDOUT:   %.loc13_10.3: init i32 = initialize_from %.loc13_10.1 to %.loc13_10.2
// CHECK:STDOUT:   %.loc13_10.4: f64 = struct_access %y.ref, member0
// CHECK:STDOUT:   %.loc13_10.5: ref f64 = struct_access %return, member0
// CHECK:STDOUT:   %.loc13_10.6: init f64 = initialize_from %.loc13_10.4 to %.loc13_10.5
// CHECK:STDOUT:   %.loc13_10.7: init {.a: i32, .b: f64} = struct_init %y.ref, (%.loc13_10.3, %.loc13_10.6)
// CHECK:STDOUT:   return %.loc13_10.7
// CHECK:STDOUT: }
