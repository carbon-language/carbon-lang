// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_address_of_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_address_of_value.carbon

fn G() -> i32;

fn H() -> {.a: i32};

fn AddressOfLiteral() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &0;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &true;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &true;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &1.0;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &1.0;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &"Hello";
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &"Hello";
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &(1, 2);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(1, 2);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &{.a = 5};
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &{.a = 5};
}

fn AddressOfOperator() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &(true and false);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(true and false);
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of a temporary object [AddrOfEphemeralRef]
  // CHECK:STDERR:   &H().a;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &H().a;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &(not true);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(not true);
}

fn AddressOfCall() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &G();
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &G();
}

fn AddressOfType() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &i32;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &i32;
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &(const i32*);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &(const i32*);
}

fn AddressOfTupleElementValue() {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:3: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   &((1, 2).0);
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  &((1, 2).0);
}

fn AddressOfParam(param: i32) {
  // CHECK:STDERR: fail_address_of_value.carbon:[[@LINE+4]]:26: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   var param_addr: i32* = &param;
  // CHECK:STDERR:                          ^
  // CHECK:STDERR:
  var param_addr: i32* = &param;
}

// CHECK:STDOUT: --- fail_address_of_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.ba9: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.268: type = pattern_type %struct_type.a.ba9 [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddressOfLiteral.type: type = fn_type @AddressOfLiteral [concrete]
// CHECK:STDOUT:   %AddressOfLiteral: %AddressOfLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.1d1: type = ptr_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %float: f64 = float_value 1 [concrete]
// CHECK:STDOUT:   %ptr.ef1: type = ptr_type f64 [concrete]
// CHECK:STDOUT:   %ptr.a45: type = ptr_type String [concrete]
// CHECK:STDOUT:   %str: String = string_literal "Hello" [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ptr.b50: type = ptr_type %tuple.type [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ptr.4e0: type = ptr_type %struct_type.a.a6c [concrete]
// CHECK:STDOUT:   %AddressOfOperator.type: type = fn_type @AddressOfOperator [concrete]
// CHECK:STDOUT:   %AddressOfOperator: %AddressOfOperator.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.c80: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%struct_type.a.ba9) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.691: %T.as.Destroy.impl.Op.type.c80 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1bb: type = ptr_type %struct_type.a.ba9 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.51b: <specific function> = specific_function %T.as.Destroy.impl.Op.691, @T.as.Destroy.impl.Op(%struct_type.a.ba9) [concrete]
// CHECK:STDOUT:   %AddressOfCall.type: type = fn_type @AddressOfCall [concrete]
// CHECK:STDOUT:   %AddressOfCall: %AddressOfCall.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddressOfType.type: type = fn_type @AddressOfType [concrete]
// CHECK:STDOUT:   %AddressOfType: %AddressOfType.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.db7: type = ptr_type type [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr.36b: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %AddressOfTupleElementValue.type: type = fn_type @AddressOfTupleElementValue [concrete]
// CHECK:STDOUT:   %AddressOfTupleElementValue: %AddressOfTupleElementValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT:   %AddressOfParam.type: type = fn_type @AddressOfParam [concrete]
// CHECK:STDOUT:   %AddressOfParam: %AddressOfParam.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %ptr.235 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.ba2: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%ptr.235) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.649: %T.as.Destroy.impl.Op.type.ba2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.5d5: type = ptr_type %ptr.235 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn.829: <specific function> = specific_function %T.as.Destroy.impl.Op.649, @T.as.Destroy.impl.Op(%ptr.235) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
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
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %return.patt: %pattern_type.268 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.268 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.ba9]
// CHECK:STDOUT:     %return.param: ref %struct_type.a.ba9 = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.a.ba9 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddressOfLiteral.decl: %AddressOfLiteral.type = fn_decl @AddressOfLiteral [concrete = constants.%AddressOfLiteral] {} {}
// CHECK:STDOUT:   %AddressOfOperator.decl: %AddressOfOperator.type = fn_decl @AddressOfOperator [concrete = constants.%AddressOfOperator] {} {}
// CHECK:STDOUT:   %AddressOfCall.decl: %AddressOfCall.type = fn_decl @AddressOfCall [concrete = constants.%AddressOfCall] {} {}
// CHECK:STDOUT:   %AddressOfType.decl: %AddressOfType.type = fn_decl @AddressOfType [concrete = constants.%AddressOfType] {} {}
// CHECK:STDOUT:   %AddressOfTupleElementValue.decl: %AddressOfTupleElementValue.type = fn_decl @AddressOfTupleElementValue [concrete = constants.%AddressOfTupleElementValue] {} {}
// CHECK:STDOUT:   %AddressOfParam.decl: %AddressOfParam.type = fn_decl @AddressOfParam [concrete = constants.%AddressOfParam] {
// CHECK:STDOUT:     %param.patt: %pattern_type.7ce = binding_pattern param [concrete]
// CHECK:STDOUT:     %param.param_patt: %pattern_type.7ce = value_param_pattern %param.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %param.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc99: type = splice_block %i32.loc99 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc99: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc99: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %param: %i32 = bind_name param, %param.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> %struct_type.a.ba9;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfLiteral() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %addr.loc24: %ptr.1d1 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %addr.loc29: %ptr.bb2 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %float: f64 = float_value 1 [concrete = constants.%float]
// CHECK:STDOUT:   %addr.loc34: %ptr.ef1 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %str: String = string_literal "Hello" [concrete = constants.%str]
// CHECK:STDOUT:   %addr.loc39: %ptr.a45 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc44: %tuple.type = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:   %addr.loc44: %ptr.b50 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc49: %struct_type.a.a6c = struct_literal (%int_5)
// CHECK:STDOUT:   %addr.loc49: %ptr.4e0 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfOperator() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true.loc57: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc57_10: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc57 br !and.rhs else br !and.result(%false.loc57_10)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %false.loc57_14: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result(%false.loc57_14)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc57: bool = block_arg !and.result [concrete = constants.%false]
// CHECK:STDOUT:   %addr.loc57: %ptr.bb2 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:   %H.call: init %struct_type.a.ba9 = call %H.ref()
// CHECK:STDOUT:   %.loc62_6.1: ref %struct_type.a.ba9 = temporary_storage
// CHECK:STDOUT:   %.loc62_6.2: ref %struct_type.a.ba9 = temporary %.loc62_6.1, %H.call
// CHECK:STDOUT:   %.loc62_7: ref %i32 = struct_access %.loc62_6.2, element0
// CHECK:STDOUT:   %addr.loc62_3: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %true.loc67: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc67: bool = not %true.loc67 [concrete = constants.%false]
// CHECK:STDOUT:   %addr.loc67: %ptr.bb2 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc62_6.1, constants.%T.as.Destroy.impl.Op.691
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.691, @T.as.Destroy.impl.Op(constants.%struct_type.a.ba9) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.51b]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc62_6.1, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc62_6: %ptr.1bb = addr_of %.loc62_6.1
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc62_6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfCall() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init %i32 = call %G.ref()
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc83: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc83: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %addr.loc83: %ptr.db7 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %int_32.loc88: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc88: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %const: type = const_type %i32.loc88 [concrete = constants.%const]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete = constants.%ptr.36b]
// CHECK:STDOUT:   %addr.loc88: %ptr.db7 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfTupleElementValue() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc96_10.1: %tuple.type = tuple_literal (%int_1, %int_2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_1, %int_2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc96_10.2: %tuple.type = converted %.loc96_10.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0: Core.IntLiteral = tuple_access %.loc96_10.2, element0 [concrete = constants.%int_1]
// CHECK:STDOUT:   %addr: %ptr.1d1 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddressOfParam(%param.param: %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %param_addr.patt: %pattern_type.fe8 = binding_pattern param_addr [concrete]
// CHECK:STDOUT:     %param_addr.var_patt: %pattern_type.fe8 = var_pattern %param_addr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %param_addr.var: ref %ptr.235 = var %param_addr.var_patt
// CHECK:STDOUT:   %param.ref: %i32 = name_ref param, %param
// CHECK:STDOUT:   %addr.loc104_26: %ptr.235 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   assign %param_addr.var, %addr.loc104_26
// CHECK:STDOUT:   %.loc104: type = splice_block %ptr [concrete = constants.%ptr.235] {
// CHECK:STDOUT:     %int_32.loc104: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc104: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr: type = ptr_type %i32.loc104 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %param_addr: ref %ptr.235 = bind_name param_addr, %param_addr.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %param_addr.var, constants.%T.as.Destroy.impl.Op.649
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.649, @T.as.Destroy.impl.Op(constants.%ptr.235) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn.829]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %param_addr.var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc104_3: %ptr.5d5 = addr_of %param_addr.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc104_3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
