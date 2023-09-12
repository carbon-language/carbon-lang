// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_address_of_value.carbon" {

fn G() -> i32;
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @G

fn H() -> {.a: i32};
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @H

fn AddressOfLiteral() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @AddressOfLiteral
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &0;
  // CHECK:STDERR:   ^
  &0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &true;
  // CHECK:STDERR:   ^
  &true;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &1.0;
  // CHECK:STDERR:   ^
  &1.0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &"Hello";
  // CHECK:STDERR:   ^
  &"Hello";
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(1, 2);
  // CHECK:STDERR:   ^
  &(1, 2);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &{.a = 5};
  // CHECK:STDERR:   ^
  &{.a = 5};
}

fn AddressOfOperator() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @AddressOfOperator
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(1 + 1);
  // CHECK:STDERR:   ^
  &(1 + 1);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of a temporary object.
  // CHECK:STDERR:   &H().a;
  // CHECK:STDERR:   ^
  &H().a;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(not true);
  // CHECK:STDERR:   ^
  &(not true);
}

fn AddressOfCall() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @AddressOfCall
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &G();
  // CHECK:STDERR:   ^
  &G();
}

fn AddressOfType() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @AddressOfType
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &i32;
  // CHECK:STDERR:   ^
  &i32;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(const i32*);
  // CHECK:STDERR:   ^
  &(const i32*);
}

fn AddressOfTupleElementValue() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @AddressOfTupleElementValue
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:3: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &((1, 2)[0]);
  // CHECK:STDERR:   ^
  &((1, 2)[0]);
}

fn AddressOfParameter(param: i32) {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @AddressOfParameter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> {.a: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-77]]_4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc[[@LINE-78]]_3.1: type = ptr_type i32
// CHECK:STDOUT:   %.loc[[@LINE-79]]_3.2: i32* = address_of %.loc[[@LINE-79]]_4
// CHECK:STDOUT:   %.loc[[@LINE-76]]_4: bool = bool_literal true
// CHECK:STDOUT:   %.loc[[@LINE-77]]_3.1: type = ptr_type bool
// CHECK:STDOUT:   %.loc[[@LINE-78]]_3.2: bool* = address_of %.loc[[@LINE-78]]_4
// CHECK:STDOUT:   %.loc[[@LINE-75]]_4: f64 = real_literal 10e-1
// CHECK:STDOUT:   %.loc[[@LINE-76]]_3.1: type = ptr_type f64
// CHECK:STDOUT:   %.loc[[@LINE-77]]_3.2: f64* = address_of %.loc[[@LINE-77]]_4
// CHECK:STDOUT:   %.loc[[@LINE-74]]_4: String = string_literal "Hello"
// CHECK:STDOUT:   %.loc[[@LINE-75]]_3.1: type = ptr_type String
// CHECK:STDOUT:   %.loc[[@LINE-76]]_3.2: String* = address_of %.loc[[@LINE-76]]_4
// CHECK:STDOUT:   %.loc[[@LINE-73]]_5.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-74]]_5.2: i32 = stub_reference %.loc[[@LINE-74]]_5.1
// CHECK:STDOUT:   %.loc[[@LINE-75]]_8.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-76]]_8.2: i32 = stub_reference %.loc[[@LINE-76]]_8.1
// CHECK:STDOUT:   %.loc[[@LINE-77]]_9.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-78]]_9.2: (i32, i32) = tuple_literal (%.loc[[@LINE-78]]_5.2, %.loc[[@LINE-78]]_8.2)
// CHECK:STDOUT:   %.loc[[@LINE-79]]_3.1: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc[[@LINE-80]]_3.2: (i32, i32)* = address_of %.loc[[@LINE-80]]_9.2
// CHECK:STDOUT:   %.loc[[@LINE-77]]_10: i32 = int_literal 5
// CHECK:STDOUT:   %.loc[[@LINE-78]]_8: i32 = stub_reference %.loc[[@LINE-78]]_10
// CHECK:STDOUT:   %.loc[[@LINE-79]]_11: {.a: i32} = struct_literal (%.loc[[@LINE-79]]_8)
// CHECK:STDOUT:   %.loc[[@LINE-80]]_3.1: type = ptr_type {.a: i32}
// CHECK:STDOUT:   %.loc[[@LINE-81]]_3.2: {.a: i32}* = address_of %.loc[[@LINE-81]]_11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfOperator() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-79]]_5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-80]]_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-81]]_7: i32 = add %.loc[[@LINE-81]]_5, %.loc[[@LINE-81]]_9
// CHECK:STDOUT:   %.loc[[@LINE-82]]_3: i32* = address_of %.loc[[@LINE-82]]_7
// CHECK:STDOUT:   %.loc[[@LINE-79]]_5.1: init {.a: i32} = call @H()
// CHECK:STDOUT:   %.loc[[@LINE-80]]_5.2: ref {.a: i32} = temporary_storage
// CHECK:STDOUT:   %.loc[[@LINE-81]]_5.3: ref {.a: i32} = temporary %.loc[[@LINE-81]]_5.2, %.loc[[@LINE-81]]_5.1
// CHECK:STDOUT:   %.loc[[@LINE-82]]_7: ref i32 = struct_access %.loc[[@LINE-82]]_5.3, member0
// CHECK:STDOUT:   %.loc[[@LINE-83]]_3: i32* = address_of %.loc[[@LINE-83]]_7
// CHECK:STDOUT:   %.loc[[@LINE-80]]_9: bool = bool_literal true
// CHECK:STDOUT:   %.loc[[@LINE-81]]_5: bool = not %.loc[[@LINE-81]]_9
// CHECK:STDOUT:   %.loc[[@LINE-82]]_3: bool* = address_of %.loc[[@LINE-82]]_5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfCall() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-80]]_5: init i32 = call @G()
// CHECK:STDOUT:   %.loc[[@LINE-81]]_3: i32* = address_of %.loc[[@LINE-81]]_5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-79]]_3.1: type = ptr_type type
// CHECK:STDOUT:   %.loc[[@LINE-80]]_3.2: type* = address_of i32
// CHECK:STDOUT:   %.loc[[@LINE-77]]_5: type = const_type i32
// CHECK:STDOUT:   %.loc[[@LINE-78]]_14: type = ptr_type const i32
// CHECK:STDOUT:   %.loc[[@LINE-79]]_3: type* = address_of %.loc[[@LINE-79]]_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfTupleElementValue() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc[[@LINE-77]]_6.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-78]]_6.2: i32 = stub_reference %.loc[[@LINE-78]]_6.1
// CHECK:STDOUT:   %.loc[[@LINE-79]]_9.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-80]]_9.2: i32 = stub_reference %.loc[[@LINE-80]]_9.1
// CHECK:STDOUT:   %.loc[[@LINE-81]]_10: (i32, i32) = tuple_literal (%.loc[[@LINE-81]]_6.2, %.loc[[@LINE-81]]_9.2)
// CHECK:STDOUT:   %.loc[[@LINE-82]]_12: i32 = int_literal 0
// CHECK:STDOUT:   %.loc[[@LINE-83]]_13: i32 = tuple_index %.loc[[@LINE-83]]_10, %.loc[[@LINE-83]]_12
// CHECK:STDOUT:   %.loc[[@LINE-84]]_3: i32* = address_of %.loc[[@LINE-84]]_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfParameter(%param: i32) {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:26: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var param_addr: i32* = &param;
  // CHECK:STDERR:                          ^
  var param_addr: i32* = &param;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_22: type = ptr_type i32
  // CHECK:STDOUT:   %param_addr: ref i32* = var "param_addr"
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_26: i32* = address_of %param
  // CHECK:STDOUT:   assign %param_addr, %.loc[[@LINE-4]]_26
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
