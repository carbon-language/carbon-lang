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
// CHECK:STDOUT:   %.loc14_27: type = struct_type {.a: i32, .b: f64}
// CHECK:STDOUT:   %MakeF64.ref: <function> = name_reference "MakeF64", file.%MakeF64
// CHECK:STDOUT:   %.loc14_44.1: init f64 = call %MakeF64.ref()
// CHECK:STDOUT:   %MakeI32.ref: <function> = name_reference "MakeI32", file.%MakeI32
// CHECK:STDOUT:   %.loc14_60.1: init i32 = call %MakeI32.ref()
// CHECK:STDOUT:   %.loc14_62.1: type = struct_type {.b: f64, .a: i32}
// CHECK:STDOUT:   %.loc14_62.2: {.b: f64, .a: i32} = struct_literal (%.loc14_44.1, %.loc14_60.1)
// CHECK:STDOUT:   %.loc14_60.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc14_60.3: ref i32 = temporary %.loc14_60.2, %.loc14_60.1
// CHECK:STDOUT:   %.loc14_60.4: i32 = bind_value %.loc14_60.3
// CHECK:STDOUT:   %.loc14_44.2: ref f64 = temporary_storage
// CHECK:STDOUT:   %.loc14_44.3: ref f64 = temporary %.loc14_44.2, %.loc14_44.1
// CHECK:STDOUT:   %.loc14_44.4: f64 = bind_value %.loc14_44.3
// CHECK:STDOUT:   %.loc14_62.3: {.a: i32, .b: f64} = struct_value %.loc14_62.2, (%.loc14_60.4, %.loc14_44.4)
// CHECK:STDOUT:   %x: {.a: i32, .b: f64} = bind_name "x", %.loc14_62.3
// CHECK:STDOUT:   %.loc15_27: type = struct_type {.b: f64, .a: i32}
// CHECK:STDOUT:   %.loc14_62.4: type = ptr_type {.b: f64, .a: i32}
// CHECK:STDOUT:   %x.ref: {.a: i32, .b: f64} = name_reference "x", %x
// CHECK:STDOUT:   %.loc15_31.1: f64 = struct_access %x.ref, member1
// CHECK:STDOUT:   %.loc15_31.2: i32 = struct_access %x.ref, member0
// CHECK:STDOUT:   %.loc15_31.3: {.b: f64, .a: i32} = struct_value %x.ref, (%.loc15_31.1, %.loc15_31.2)
// CHECK:STDOUT:   %y: {.b: f64, .a: i32} = bind_name "y", %.loc15_31.3
// CHECK:STDOUT:   %y.ref: {.b: f64, .a: i32} = name_reference "y", %y
// CHECK:STDOUT:   %.loc16_10.1: i32 = struct_access %y.ref, member1
// CHECK:STDOUT:   %.loc16_10.2: ref i32 = struct_access %return, member1
// CHECK:STDOUT:   %.loc16_10.3: init i32 = initialize_from %.loc16_10.1 to %.loc16_10.2
// CHECK:STDOUT:   %.loc16_10.4: f64 = struct_access %y.ref, member0
// CHECK:STDOUT:   %.loc16_10.5: ref f64 = struct_access %return, member0
// CHECK:STDOUT:   %.loc16_10.6: init f64 = initialize_from %.loc16_10.4 to %.loc16_10.5
// CHECK:STDOUT:   %.loc16_10.7: init {.a: i32, .b: f64} = struct_init %y.ref, (%.loc16_10.3, %.loc16_10.6)
// CHECK:STDOUT:   return %.loc16_10.7
// CHECK:STDOUT: }
