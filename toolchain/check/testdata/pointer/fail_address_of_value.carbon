// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_address_of_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_address_of_value.carbon

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
  // CHECK:STDERR:   &((1, 2).0);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &((1, 2).0);
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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %AddressOfLiteral.type: type = fn_type @AddressOfLiteral [template]
// CHECK:STDOUT:   %AddressOfLiteral: %AddressOfLiteral.type = struct_value () [template]
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
// CHECK:STDOUT:   %.14: type = ptr_type %.13 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.16: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %AddressOfOperator.type: type = fn_type @AddressOfOperator [template]
// CHECK:STDOUT:   %AddressOfOperator: %AddressOfOperator.type = struct_value () [template]
// CHECK:STDOUT:   %.17: bool = bool_literal false [template]
// CHECK:STDOUT:   %AddressOfCall.type: type = fn_type @AddressOfCall [template]
// CHECK:STDOUT:   %AddressOfCall: %AddressOfCall.type = struct_value () [template]
// CHECK:STDOUT:   %AddressOfType.type: type = fn_type @AddressOfType [template]
// CHECK:STDOUT:   %AddressOfType: %AddressOfType.type = struct_value () [template]
// CHECK:STDOUT:   %.18: type = ptr_type type [template]
// CHECK:STDOUT:   %.19: type = const_type i32 [template]
// CHECK:STDOUT:   %.20: type = ptr_type %.19 [template]
// CHECK:STDOUT:   %AddressOfTupleElementValue.type: type = fn_type @AddressOfTupleElementValue [template]
// CHECK:STDOUT:   %AddressOfTupleElementValue: %AddressOfTupleElementValue.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.13 = tuple_value (%.11, %.12) [template]
// CHECK:STDOUT:   %AddressOfParam.type: type = fn_type @AddressOfParam [template]
// CHECK:STDOUT:   %AddressOfParam: %AddressOfParam.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .AddressOfLiteral = %AddressOfLiteral.decl
// CHECK:STDOUT:     .AddressOfOperator = %AddressOfOperator.decl
// CHECK:STDOUT:     .AddressOfCall = %AddressOfCall.decl
// CHECK:STDOUT:     .AddressOfType = %AddressOfType.decl
// CHECK:STDOUT:     .AddressOfTupleElementValue = %AddressOfTupleElementValue.decl
// CHECK:STDOUT:     .AddressOfParam = %AddressOfParam.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_16.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_16.2: type = converted %int.make_type_32.loc13, %.loc13_16.1 [template = i32]
// CHECK:STDOUT:     %.loc13_19: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:     @H.%return: ref %.2 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddressOfLiteral.decl: %AddressOfLiteral.type = fn_decl @AddressOfLiteral [template = constants.%AddressOfLiteral] {}
// CHECK:STDOUT:   %AddressOfOperator.decl: %AddressOfOperator.type = fn_decl @AddressOfOperator [template = constants.%AddressOfOperator] {}
// CHECK:STDOUT:   %AddressOfCall.decl: %AddressOfCall.type = fn_decl @AddressOfCall [template = constants.%AddressOfCall] {}
// CHECK:STDOUT:   %AddressOfType.decl: %AddressOfType.type = fn_decl @AddressOfType [template = constants.%AddressOfType] {}
// CHECK:STDOUT:   %AddressOfTupleElementValue.decl: %AddressOfTupleElementValue.type = fn_decl @AddressOfTupleElementValue [template = constants.%AddressOfTupleElementValue] {}
// CHECK:STDOUT:   %AddressOfParam.decl: %AddressOfParam.type = fn_decl @AddressOfParam [template = constants.%AddressOfParam] {
// CHECK:STDOUT:     %int.make_type_32.loc95: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc95_26.1: type = value_of_initializer %int.make_type_32.loc95 [template = i32]
// CHECK:STDOUT:     %.loc95_26.2: type = converted %int.make_type_32.loc95, %.loc95_26.1 [template = i32]
// CHECK:STDOUT:     %param.loc95_19.1: i32 = param param
// CHECK:STDOUT:     @AddressOfParam.%param: i32 = bind_name param, %param.loc95_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> %.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20_4: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc20_3: %.4 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc25_4: bool = bool_literal true [template = constants.%.5]
// CHECK:STDOUT:   %.loc25_3: %.6 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc30_4: f64 = float_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc30_3: %.8 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc35_4: String = string_literal "Hello" [template = constants.%.10]
// CHECK:STDOUT:   %.loc35_3: %.9 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc40_5: i32 = int_literal 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc40_8: i32 = int_literal 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc40_9: %.13 = tuple_literal (%.loc40_5, %.loc40_8)
// CHECK:STDOUT:   %.loc40_3: %.14 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc45_10: i32 = int_literal 5 [template = constants.%.15]
// CHECK:STDOUT:   %.loc45_11: %.2 = struct_literal (%.loc45_10)
// CHECK:STDOUT:   %.loc45_3: %.16 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfOperator() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc53_5: bool = bool_literal true [template = constants.%.5]
// CHECK:STDOUT:   %.loc53_10.1: bool = bool_literal false [template = constants.%.17]
// CHECK:STDOUT:   if %.loc53_5 br !and.rhs else br !and.result(%.loc53_10.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %.loc53_14: bool = bool_literal false [template = constants.%.17]
// CHECK:STDOUT:   br !and.result(%.loc53_14)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc53_10.2: bool = block_arg !and.result [template = constants.%.17]
// CHECK:STDOUT:   %.loc53_3: %.6 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [template = constants.%H]
// CHECK:STDOUT:   %H.call: init %.2 = call %H.ref()
// CHECK:STDOUT:   %.loc58_5.1: ref %.2 = temporary_storage
// CHECK:STDOUT:   %.loc58_5.2: ref %.2 = temporary %.loc58_5.1, %H.call
// CHECK:STDOUT:   %.loc58_7: ref i32 = struct_access %.loc58_5.2, element0
// CHECK:STDOUT:   %.loc58_3: %.4 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %.loc63_9: bool = bool_literal true [template = constants.%.5]
// CHECK:STDOUT:   %.loc63_5: bool = not %.loc63_9 [template = constants.%.17]
// CHECK:STDOUT:   %.loc63_3: %.6 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfCall() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.call: init i32 = call %G.ref()
// CHECK:STDOUT:   %.loc71: %.4 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc79: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc79: %.18 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   %int.make_type_32.loc84: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc84_5.1: type = value_of_initializer %int.make_type_32.loc84 [template = i32]
// CHECK:STDOUT:   %.loc84_5.2: type = converted %int.make_type_32.loc84, %.loc84_5.1 [template = i32]
// CHECK:STDOUT:   %.loc84_5.3: type = const_type i32 [template = constants.%.19]
// CHECK:STDOUT:   %.loc84_14: type = ptr_type %.19 [template = constants.%.20]
// CHECK:STDOUT:   %.loc84_3: %.18 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfTupleElementValue() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc92_6: i32 = int_literal 1 [template = constants.%.11]
// CHECK:STDOUT:   %.loc92_9: i32 = int_literal 2 [template = constants.%.12]
// CHECK:STDOUT:   %.loc92_10.1: %.13 = tuple_literal (%.loc92_6, %.loc92_9)
// CHECK:STDOUT:   %.loc92_12: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %tuple: %.13 = tuple_value (%.loc92_6, %.loc92_9) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc92_10.2: %.13 = converted %.loc92_10.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %.loc92_11: i32 = tuple_index %.loc92_10.2, %.loc92_12 [template = constants.%.11]
// CHECK:STDOUT:   %.loc92_3: %.4 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfParam(%param: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc99_22.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc99_22.2: type = converted %int.make_type_32, %.loc99_22.1 [template = i32]
// CHECK:STDOUT:   %.loc99_22.3: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:   %param_addr.var: ref %.4 = var param_addr
// CHECK:STDOUT:   %param_addr: ref %.4 = bind_name param_addr, %param_addr.var
// CHECK:STDOUT:   %param.ref: i32 = name_ref param, %param
// CHECK:STDOUT:   %.loc99_26: %.4 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %param_addr.var, %.loc99_26
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
