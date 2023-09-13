// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32);

fn G() -> (i32, i32) {
  var v: (i32, i32) = F();
  v = F();
  return F();
}

fn H() -> i32 {
  return G()[0];
}

// CHECK:STDOUT: file "in_place_tuple_initialization.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc9 = fn_decl @G
// CHECK:STDOUT:   %.loc15 = fn_decl @H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_11: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_16: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_19: (type, type) = tuple_literal (%.loc10_11, %.loc10_16)
// CHECK:STDOUT:   %v: ref (i32, i32) = var "v"
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc10_24: init (i32, i32) = call @F() to %v
// CHECK:STDOUT:   assign %v, %.loc10_24
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc11: init (i32, i32) = call @F() to %v
// CHECK:STDOUT:   assign %v, %.loc11
// CHECK:STDOUT:   no_op
// CHECK:STDOUT:   %.loc12: init (i32, i32) = call @F() to %return
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_11.1: ref (i32, i32) = temporary_storage
// CHECK:STDOUT:   %.loc16_11.2: init (i32, i32) = call @G() to %.loc16_11.1
// CHECK:STDOUT:   %.loc16_14: i32 = int_literal 0
// CHECK:STDOUT:   %.loc16_11.3: ref (i32, i32) = temporary %.loc16_11.1, %.loc16_11.2
// CHECK:STDOUT:   %.loc16_15.1: ref i32 = tuple_index %.loc16_11.3, %.loc16_14
// CHECK:STDOUT:   %.loc16_15.2: i32 = bind_value %.loc16_15.1
// CHECK:STDOUT:   return %.loc16_15.2
// CHECK:STDOUT: }
