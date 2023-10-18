// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G() -> i32;

fn H() -> {.a: i32};

fn AddressOfLiteral() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &0;
  // CHECK:STDERR:   ^
  &0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &true;
  // CHECK:STDERR:   ^
  &true;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &1.0;
  // CHECK:STDERR:   ^
  &1.0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &"Hello";
  // CHECK:STDERR:   ^
  &"Hello";
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(1, 2);
  // CHECK:STDERR:   ^
  &(1, 2);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &{.a = 5};
  // CHECK:STDERR:   ^
  &{.a = 5};
}

fn AddressOfOperator() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(1 + 1);
  // CHECK:STDERR:   ^
  &(1 + 1);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of a temporary object.
  // CHECK:STDERR:   &H().a;
  // CHECK:STDERR:   ^
  &H().a;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(not true);
  // CHECK:STDERR:   ^
  &(not true);
}

fn AddressOfCall() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &G();
  // CHECK:STDERR:   ^
  &G();
}

fn AddressOfType() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &i32;
  // CHECK:STDERR:   ^
  &i32;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(const i32*);
  // CHECK:STDERR:   ^
  &(const i32*);
}

fn AddressOfTupleElementValue() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &((1, 2)[0]);
  // CHECK:STDERR:   ^
  &((1, 2)[0]);
}

fn AddressOfParameter(param: i32) {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:26: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var param_addr: i32* = &param;
  // CHECK:STDERR:                          ^
  var param_addr: i32* = &param;
}

// CHECK:STDOUT: file "fail_address_of_value.carbon" {
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %H: <function> = fn_decl @H
// CHECK:STDOUT:   %AddressOfLiteral: <function> = fn_decl @AddressOfLiteral
// CHECK:STDOUT:   %AddressOfOperator: <function> = fn_decl @AddressOfOperator
// CHECK:STDOUT:   %AddressOfCall: <function> = fn_decl @AddressOfCall
// CHECK:STDOUT:   %AddressOfType: <function> = fn_decl @AddressOfType
// CHECK:STDOUT:   %AddressOfTupleElementValue: <function> = fn_decl @AddressOfTupleElementValue
// CHECK:STDOUT:   %AddressOfParameter: <function> = fn_decl @AddressOfParameter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> {.a: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc15_3.1: type = ptr_type i32
// CHECK:STDOUT:   %.loc15_3.2: i32* = address_of %.loc15_4
// CHECK:STDOUT:   %.loc19_4: bool = bool_literal true
// CHECK:STDOUT:   %.loc19_3.1: type = ptr_type bool
// CHECK:STDOUT:   %.loc19_3.2: bool* = address_of %.loc19_4
// CHECK:STDOUT:   %.loc23_4: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc23_3.1: type = ptr_type f64
// CHECK:STDOUT:   %.loc23_3.2: f64* = address_of %.loc23_4
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT:   %.loc27_4: String = string_literal "Hello"
// CHECK:STDOUT:   %.loc27_3: String* = address_of %.loc27_4
// CHECK:STDOUT:   %.loc31_5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc31_8: i32 = int_literal 2
// CHECK:STDOUT:   %.loc31_9.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc31_9.2: (i32, i32) = tuple_literal (%.loc31_5, %.loc31_8)
// CHECK:STDOUT:   %.loc31_3.1: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc31_3.2: (i32, i32)* = address_of %.loc31_9.2
// CHECK:STDOUT:   %.loc35_10: i32 = int_literal 5
// CHECK:STDOUT:   %.loc35_11: {.a: i32} = struct_literal (%.loc35_10)
// CHECK:STDOUT:   %.loc35_3.1: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc35_3.2: {.a: i32}* = address_of %.loc35_11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfOperator() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc42_5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc42_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc42_7: i32 = add %.loc42_5, %.loc42_9
// CHECK:STDOUT:   %.loc42_3: i32* = address_of %.loc42_7
// CHECK:STDOUT:   %H.ref: <function> = name_reference "H", package.%H
// CHECK:STDOUT:   %.loc46_5.1: init {.a: i32} = call %H.ref()
// CHECK:STDOUT:   %.loc46_5.2: ref {.a: i32} = temporary_storage
// CHECK:STDOUT:   %.loc46_5.3: ref {.a: i32} = temporary %.loc46_5.2, %.loc46_5.1
// CHECK:STDOUT:   %.loc46_7: ref i32 = struct_access %.loc46_5.3, member0
// CHECK:STDOUT:   %.loc46_3: i32* = address_of %.loc46_7
// CHECK:STDOUT:   %.loc50_9: bool = bool_literal true
// CHECK:STDOUT:   %.loc50_5: bool = not %.loc50_9
// CHECK:STDOUT:   %.loc50_3: bool* = address_of %.loc50_5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfCall() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: <function> = name_reference "G", package.%G
// CHECK:STDOUT:   %.loc57_5: init i32 = call %G.ref()
// CHECK:STDOUT:   %.loc57_3: i32* = address_of %.loc57_5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc64_3.1: type = ptr_type type
// CHECK:STDOUT:   %.loc64_3.2: type* = address_of i32
// CHECK:STDOUT:   %.loc68_5: type = const_type i32
// CHECK:STDOUT:   %.loc68_14: type = ptr_type const i32
// CHECK:STDOUT:   %.loc68_3: type* = address_of %.loc68_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfTupleElementValue() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc75_6: i32 = int_literal 1
// CHECK:STDOUT:   %.loc75_9: i32 = int_literal 2
// CHECK:STDOUT:   %.loc75_10.1: (i32, i32) = tuple_literal (%.loc75_6, %.loc75_9)
// CHECK:STDOUT:   %.loc75_12: i32 = int_literal 0
// CHECK:STDOUT:   %.loc75_10.2: (i32, i32) = tuple_value %.loc75_10.1, (%.loc75_6, %.loc75_9)
// CHECK:STDOUT:   %.loc75_13: i32 = tuple_index %.loc75_10.2, %.loc75_12
// CHECK:STDOUT:   %.loc75_3: i32* = address_of %.loc75_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfParameter(%param: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc82_22: type = ptr_type i32
// CHECK:STDOUT:   %param_addr: ref i32* = var "param_addr"
// CHECK:STDOUT:   %param.ref: i32 = name_reference "param", %param
// CHECK:STDOUT:   %.loc82_26: i32* = address_of %param.ref
// CHECK:STDOUT:   assign %param_addr, %.loc82_26
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
