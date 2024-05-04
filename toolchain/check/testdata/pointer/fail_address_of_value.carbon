// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G() -> i32;

fn H() -> {.a: i32};

fn AddressOfLiteral() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &0;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &true;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &true;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &1.0;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &1.0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &"Hello";
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &"Hello";
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(1, 2);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(1, 2);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &{.a = 5};
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &{.a = 5};
}

fn AddressOfOperator() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(true and false);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(true and false);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of a temporary object.
  // CHECK:STDERR:   &H().a;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &H().a;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(not true);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(not true);
}

fn AddressOfCall() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &G();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &G();
}

fn AddressOfType() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &i32;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &i32;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &(const i32*);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(const i32*);
}

fn AddressOfTupleElementValue() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   &((1, 2)[0]);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &((1, 2)[0]);
}

fn AddressOfParam(param: i32) {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+3]]:26: ERROR: Cannot take the address of non-reference expression.
  // CHECK:STDERR:   var param_addr: i32* = &param;
  // CHECK:STDERR:                          ^
  var param_addr: i32* = &param;
}

// CHECK:STDOUT: --- fail_address_of_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: G = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.2: H = struct_value () [template]
// CHECK:STDOUT:   %AddressOfLiteral: type = fn_type @AddressOfLiteral [template]
// CHECK:STDOUT:   %struct.3: AddressOfLiteral = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.5: bool = bool_literal true [template]
// CHECK:STDOUT:   %.6: type = ptr_type bool [template]
// CHECK:STDOUT:   %.7: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.8: type = ptr_type f64 [template]
// CHECK:STDOUT:   %.9: type = ptr_type String [template]
// CHECK:STDOUT:   %.10: String = string_literal "Hello" [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.13: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.14: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.16: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %AddressOfOperator: type = fn_type @AddressOfOperator [template]
// CHECK:STDOUT:   %struct.4: AddressOfOperator = struct_value () [template]
// CHECK:STDOUT:   %.17: bool = bool_literal false [template]
// CHECK:STDOUT:   %AddressOfCall: type = fn_type @AddressOfCall [template]
// CHECK:STDOUT:   %struct.5: AddressOfCall = struct_value () [template]
// CHECK:STDOUT:   %AddressOfType: type = fn_type @AddressOfType [template]
// CHECK:STDOUT:   %struct.6: AddressOfType = struct_value () [template]
// CHECK:STDOUT:   %.18: type = ptr_type type [template]
// CHECK:STDOUT:   %.19: type = const_type i32 [template]
// CHECK:STDOUT:   %.20: type = ptr_type const i32 [template]
// CHECK:STDOUT:   %AddressOfTupleElementValue: type = fn_type @AddressOfTupleElementValue [template]
// CHECK:STDOUT:   %struct.7: AddressOfTupleElementValue = struct_value () [template]
// CHECK:STDOUT:   %tuple: (i32, i32) = tuple_value (%.11, %.12) [template]
// CHECK:STDOUT:   %AddressOfParam: type = fn_type @AddressOfParam [template]
// CHECK:STDOUT:   %struct.8: AddressOfParam = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .AddressOfLiteral = %AddressOfLiteral.decl
// CHECK:STDOUT:     .AddressOfOperator = %AddressOfOperator.decl
// CHECK:STDOUT:     .AddressOfCall = %AddressOfCall.decl
// CHECK:STDOUT:     .AddressOfType = %AddressOfType.decl
// CHECK:STDOUT:     .AddressOfTupleElementValue = %AddressOfTupleElementValue.decl
// CHECK:STDOUT:     .AddressOfParam = %AddressOfParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.1] {
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.2] {
// CHECK:STDOUT:     %.loc9: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:     @H.%return: ref {.a: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddressOfLiteral.decl: AddressOfLiteral = fn_decl @AddressOfLiteral [template = constants.%struct.3] {}
// CHECK:STDOUT:   %AddressOfOperator.decl: AddressOfOperator = fn_decl @AddressOfOperator [template = constants.%struct.4] {}
// CHECK:STDOUT:   %AddressOfCall.decl: AddressOfCall = fn_decl @AddressOfCall [template = constants.%struct.5] {}
// CHECK:STDOUT:   %AddressOfType.decl: AddressOfType = fn_decl @AddressOfType [template = constants.%struct.6] {}
// CHECK:STDOUT:   %AddressOfTupleElementValue.decl: AddressOfTupleElementValue = fn_decl @AddressOfTupleElementValue [template = constants.%struct.7] {}
// CHECK:STDOUT:   %AddressOfParam.decl: AddressOfParam = fn_decl @AddressOfParam [template = constants.%struct.8] {
// CHECK:STDOUT:     %param.loc91_19.1: i32 = param param
// CHECK:STDOUT:     @AddressOfParam.%param: i32 = bind_name param, %param.loc91_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> {.a: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_4: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc16_3: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc21_4: bool = bool_literal true [template = constants.%.5]
// CHECK:STDOUT:   %.loc21_3: bool* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc26_4: f64 = float_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc26_3: f64* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc31_4: String = string_literal "Hello" [template = constants.%.10]
// CHECK:STDOUT:   %.loc31_3: String* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc36_5: i32 = int_literal 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc36_8: i32 = int_literal 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc36_9: (i32, i32) = tuple_literal (%.loc36_5, %.loc36_8)
// CHECK:STDOUT:   %.loc36_3: (i32, i32)* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc41_10: i32 = int_literal 5 [template = constants.%.15]
// CHECK:STDOUT:   %.loc41_11: {.a: i32} = struct_literal (%.loc41_10)
// CHECK:STDOUT:   %.loc41_3: {.a: i32}* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfOperator() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc49_5: bool = bool_literal true [template = constants.%.5]
// CHECK:STDOUT:   %.loc49_10.1: bool = bool_literal false [template = constants.%.17]
// CHECK:STDOUT:   if %.loc49_5 br !and.rhs else br !and.result(%.loc49_10.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %.loc49_14: bool = bool_literal false [template = constants.%.17]
// CHECK:STDOUT:   br !and.result(%.loc49_14)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc49_10.2: bool = block_arg !and.result [template = constants.%.17]
// CHECK:STDOUT:   %.loc49_3: bool* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %H.ref: H = name_ref H, file.%H.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %H.call: init {.a: i32} = call %H.ref()
// CHECK:STDOUT:   %.loc54_5.1: ref {.a: i32} = temporary_storage
// CHECK:STDOUT:   %.loc54_5.2: ref {.a: i32} = temporary %.loc54_5.1, %H.call
// CHECK:STDOUT:   %.loc54_7: ref i32 = struct_access %.loc54_5.2, element0
// CHECK:STDOUT:   %.loc54_3: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc59_9: bool = bool_literal true [template = constants.%.5]
// CHECK:STDOUT:   %.loc59_5: bool = not %.loc59_9 [template = constants.%.17]
// CHECK:STDOUT:   %.loc59_3: bool* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfCall() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %G.call: init i32 = call %G.ref()
// CHECK:STDOUT:   %.loc67: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc75: type* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc80_5: type = const_type i32 [template = constants.%.19]
// CHECK:STDOUT:   %.loc80_14: type = ptr_type const i32 [template = constants.%.20]
// CHECK:STDOUT:   %.loc80_3: type* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfTupleElementValue() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc88_6: i32 = int_literal 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc88_9: i32 = int_literal 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc88_10.1: (i32, i32) = tuple_literal (%.loc88_6, %.loc88_9)
// CHECK:STDOUT:   %.loc88_12: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %tuple: (i32, i32) = tuple_value (%.loc88_6, %.loc88_9) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc88_10.2: (i32, i32) = converted %.loc88_10.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %.loc88_13: i32 = tuple_index %.loc88_10.2, %.loc88_12 [template = constants.%.11]
// CHECK:STDOUT:   %.loc88_3: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfParam(%param: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc95_22: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %param_addr.var: ref i32* = var param_addr
// CHECK:STDOUT:   %param_addr: ref i32* = bind_name param_addr, %param_addr.var
// CHECK:STDOUT:   %param.ref: i32 = name_ref param, %param
// CHECK:STDOUT:   %.loc95_26: i32* = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %param_addr.var, %.loc95_26
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
