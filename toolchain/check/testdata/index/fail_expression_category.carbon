// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  // Indexing an array value gives a value.
  // CHECK:STDERR: fail_expression_category.carbon:[[@LINE+3]]:18: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var pb: i32* = &b[0];
  // CHECK:STDERR:                  ^
  var pb: i32* = &b[0];
  // CHECK:STDERR: fail_expression_category.carbon:[[@LINE+3]]:6: ERROR: Expression is not assignable.
  // CHECK:STDERR:   b[0] = 4;
  // CHECK:STDERR:      ^
  b[0] = 4;

  // Indexing an ephemeral reference (materialized from an initializing
  // expression) gives a value.
  // CHECK:STDERR: fail_expression_category.carbon:[[@LINE+3]]:18: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var pf: i32* = &F()[0];
  // CHECK:STDERR:                  ^
  var pf: i32* = &F()[0];
  // CHECK:STDERR: fail_expression_category.carbon:[[@LINE+3]]:8: ERROR: Expression is not assignable.
  // CHECK:STDERR:   F()[0] = 4;
  // CHECK:STDERR:        ^
  F()[0] = 4;
}

// CHECK:STDOUT: file "fail_expression_category.carbon" {
// CHECK:STDOUT:   %.loc7: type = ptr_type [i32; 3]
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: [i32; 3];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: [i32; 3]) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_14: type = ptr_type i32
// CHECK:STDOUT:   %pb: ref i32* = var "pb"
// CHECK:STDOUT:   %b.ref.loc14: [i32; 3] = name_reference "b", %b
// CHECK:STDOUT:   %.loc14_21: i32 = int_literal 0
// CHECK:STDOUT:   %.loc14_22.1: ref [i32; 3] = value_as_reference %b.ref.loc14
// CHECK:STDOUT:   %.loc14_22.2: ref i32 = array_index %.loc14_22.1, %.loc14_21
// CHECK:STDOUT:   %.loc14_22.3: i32 = bind_value %.loc14_22.2
// CHECK:STDOUT:   %.loc14_18: i32* = address_of %.loc14_22.3
// CHECK:STDOUT:   assign %pb, %.loc14_18
// CHECK:STDOUT:   %b.ref.loc18: [i32; 3] = name_reference "b", %b
// CHECK:STDOUT:   %.loc18_5: i32 = int_literal 0
// CHECK:STDOUT:   %.loc18_6.1: ref [i32; 3] = value_as_reference %b.ref.loc18
// CHECK:STDOUT:   %.loc18_6.2: ref i32 = array_index %.loc18_6.1, %.loc18_5
// CHECK:STDOUT:   %.loc18_6.3: i32 = bind_value %.loc18_6.2
// CHECK:STDOUT:   %.loc18_10: i32 = int_literal 4
// CHECK:STDOUT:   assign %.loc18_6.3, %.loc18_10
// CHECK:STDOUT:   %.loc25_14: type = ptr_type i32
// CHECK:STDOUT:   %pf: ref i32* = var "pf"
// CHECK:STDOUT:   %F.ref.loc25: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc25_20.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc25_20.2: init [i32; 3] = call %F.ref.loc25() to %.loc25_20.1
// CHECK:STDOUT:   %.loc25_23: i32 = int_literal 0
// CHECK:STDOUT:   %.loc25_20.3: ref [i32; 3] = temporary %.loc25_20.1, %.loc25_20.2
// CHECK:STDOUT:   %.loc25_24.1: ref i32 = array_index %.loc25_20.3, %.loc25_23
// CHECK:STDOUT:   %.loc25_24.2: i32 = bind_value %.loc25_24.1
// CHECK:STDOUT:   %.loc25_18: i32* = address_of %.loc25_24.2
// CHECK:STDOUT:   assign %pf, %.loc25_18
// CHECK:STDOUT:   %F.ref.loc29: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc29_4.1: ref [i32; 3] = temporary_storage
// CHECK:STDOUT:   %.loc29_4.2: init [i32; 3] = call %F.ref.loc29() to %.loc29_4.1
// CHECK:STDOUT:   %.loc29_7: i32 = int_literal 0
// CHECK:STDOUT:   %.loc29_4.3: ref [i32; 3] = temporary %.loc29_4.1, %.loc29_4.2
// CHECK:STDOUT:   %.loc29_8.1: ref i32 = array_index %.loc29_4.3, %.loc29_7
// CHECK:STDOUT:   %.loc29_8.2: i32 = bind_value %.loc29_8.1
// CHECK:STDOUT:   %.loc29_12: i32 = int_literal 4
// CHECK:STDOUT:   assign %.loc29_8.2, %.loc29_12
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
