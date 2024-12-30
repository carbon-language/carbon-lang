// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/greater_eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/greater_eq.carbon

// --- int_greater_eq.carbon

library "[[@TEST_NAME]]";

fn GreaterEq(a: i32, b: i32) -> bool = "int.greater_eq";
fn Negate(a: i32) -> i32 = "int.snegate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if GreaterEq(1, 2) then True else False);
  true_ as (if GreaterEq(1, 1) then True else False);
  true_ as (if GreaterEq(1, 0) then True else False);
  false_ as (if GreaterEq(Negate(1), 0) then True else False);
  true_ as (if GreaterEq(0, Negate(1)) then True else False);
}

fn RuntimeCall(a: i32, b: i32) -> bool {
  return GreaterEq(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn GreaterEq(a: Core.IntLiteral(), b: Core.IntLiteral()) -> bool = "int.greater_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(GreaterEq(5, 5)) as Expect(true);
  Test(GreaterEq(5, 6)) as Expect(false);
  Test(GreaterEq(6, 5)) as Expect(true);
  Test(GreaterEq(-1, -1)) as Expect(true);
  Test(GreaterEq(-1, 1)) as Expect(false);
  Test(GreaterEq(1, -1)) as Expect(true);
}

// --- mixed.carbon

library "[[@TEST_NAME]]";

fn GreaterEq(a: Core.IntLiteral(), b: i32) -> bool = "int.greater_eq";

class Expect(B:! bool) {}
fn Test(B:! bool) -> Expect(B) { return {}; }

fn F() {
  Test(GreaterEq(5, 5)) as Expect(true);
  Test(GreaterEq(5, 6)) as Expect(false);
  Test(GreaterEq(6, 5)) as Expect(true);
  Test(GreaterEq(-1, -1)) as Expect(true);
  Test(GreaterEq(-1, 1)) as Expect(false);
  Test(GreaterEq(1, -1)) as Expect(true);
}

// CHECK:STDOUT: --- int_greater_eq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %GreaterEq.type: type = fn_type @GreaterEq [template]
// CHECK:STDOUT:   %GreaterEq: %GreaterEq.type = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .GreaterEq = %GreaterEq.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %GreaterEq.decl: %GreaterEq.type = fn_decl @GreaterEq [template = constants.%GreaterEq] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_33.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_33.2: type = converted %bool.make_type, %.loc4_33.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_17: type = splice_block %i32.loc4_17 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_25: type = splice_block %i32.loc4_25 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %i32.loc5_14 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {} {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %true_.patt: %True = binding_pattern true_
// CHECK:STDOUT:     %true_.param_patt: %True = value_param_pattern %true_.patt, runtime_param0
// CHECK:STDOUT:     %false_.patt: %False = binding_pattern false_
// CHECK:STDOUT:     %false_.param_patt: %False = value_param_pattern %false_.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %true_.param: %True = value_param runtime_param0
// CHECK:STDOUT:     %True.ref.loc10: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_: %True = bind_name true_, %true_.param
// CHECK:STDOUT:     %false_.param: %False = value_param runtime_param1
// CHECK:STDOUT:     %False.ref.loc10: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:     %false_: %False = bind_name false_, %false_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc18_35.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc18_35.2: type = converted %bool.make_type, %.loc18_35.1 [template = bool]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %.loc18_19: type = splice_block %i32.loc18_19 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc18_27: type = splice_block %i32.loc18_27 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc18_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc18_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @True {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterEq(%a.param_patt: %i32, %b.param_patt: %i32) -> bool = "int.greater_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_.param_patt: %True, %false_.param_patt: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false_.ref.loc11: %False = name_ref false_, %false_
// CHECK:STDOUT:   %GreaterEq.ref.loc11: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc11_27: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_27: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_27 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27: <specific function> = specific_function %Convert.bound.loc11_27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_27: init %i32 = call %Convert.specific_fn.loc11_27(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_27.1: %i32 = value_of_initializer %int.convert_checked.loc11_27 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_27.2: %i32 = converted %int_1.loc11, %.loc11_27.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_30: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_30: <bound method> = bound_method %int_2, %impl.elem0.loc11_30 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_30: <specific function> = specific_function %Convert.bound.loc11_30, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_30: init %i32 = call %Convert.specific_fn.loc11_30(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_30.1: %i32 = value_of_initializer %int.convert_checked.loc11_30 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_30.2: %i32 = converted %int_2, %.loc11_30.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %int.greater_eq.loc11: init bool = call %GreaterEq.ref.loc11(%.loc11_27.2, %.loc11_30.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %int.greater_eq.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %int.greater_eq.loc11, %.loc11_14.1 [template = constants.%false]
// CHECK:STDOUT:   if %.loc11_14.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %True.ref.loc11: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc11(%True.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %False.ref.loc11: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc11(%False.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_14.3: type = block_arg !if.expr.result.loc11 [template = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc12: %True = name_ref true_, %true_
// CHECK:STDOUT:   %GreaterEq.ref.loc12: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc12_26: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc12_29: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc12_26: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_26: <bound method> = bound_method %int_1.loc12_26, %impl.elem0.loc12_26 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_26: <specific function> = specific_function %Convert.bound.loc12_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_26: init %i32 = call %Convert.specific_fn.loc12_26(%int_1.loc12_26) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_26.1: %i32 = value_of_initializer %int.convert_checked.loc12_26 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_26.2: %i32 = converted %int_1.loc12_26, %.loc12_26.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_29: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_29: <bound method> = bound_method %int_1.loc12_29, %impl.elem0.loc12_29 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_29: <specific function> = specific_function %Convert.bound.loc12_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_29: init %i32 = call %Convert.specific_fn.loc12_29(%int_1.loc12_29) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_29.1: %i32 = value_of_initializer %int.convert_checked.loc12_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_29.2: %i32 = converted %int_1.loc12_29, %.loc12_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.greater_eq.loc12: init bool = call %GreaterEq.ref.loc12(%.loc12_26.2, %.loc12_29.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc12_13.1: bool = value_of_initializer %int.greater_eq.loc12 [template = constants.%true]
// CHECK:STDOUT:   %.loc12_13.2: bool = converted %int.greater_eq.loc12, %.loc12_13.1 [template = constants.%true]
// CHECK:STDOUT:   if %.loc12_13.2 br !if.expr.then.loc12 else br !if.expr.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12:
// CHECK:STDOUT:   %True.ref.loc12: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc12(%True.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12:
// CHECK:STDOUT:   %False.ref.loc12: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc12(%False.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12:
// CHECK:STDOUT:   %.loc12_13.3: type = block_arg !if.expr.result.loc12 [template = constants.%True]
// CHECK:STDOUT:   %true_.ref.loc13: %True = name_ref true_, %true_
// CHECK:STDOUT:   %GreaterEq.ref.loc13: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_0.loc13: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc13_26: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_26: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_26 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_26: <specific function> = specific_function %Convert.bound.loc13_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_26: init %i32 = call %Convert.specific_fn.loc13_26(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_26.1: %i32 = value_of_initializer %int.convert_checked.loc13_26 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_26.2: %i32 = converted %int_1.loc13, %.loc13_26.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_29: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13_29: <bound method> = bound_method %int_0.loc13, %impl.elem0.loc13_29 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_29: <specific function> = specific_function %Convert.bound.loc13_29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_29: init %i32 = call %Convert.specific_fn.loc13_29(%int_0.loc13) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_29.1: %i32 = value_of_initializer %int.convert_checked.loc13_29 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_29.2: %i32 = converted %int_0.loc13, %.loc13_29.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.greater_eq.loc13: init bool = call %GreaterEq.ref.loc13(%.loc13_26.2, %.loc13_29.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc13_13.1: bool = value_of_initializer %int.greater_eq.loc13 [template = constants.%true]
// CHECK:STDOUT:   %.loc13_13.2: bool = converted %int.greater_eq.loc13, %.loc13_13.1 [template = constants.%true]
// CHECK:STDOUT:   if %.loc13_13.2 br !if.expr.then.loc13 else br !if.expr.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc13:
// CHECK:STDOUT:   %True.ref.loc13: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc13(%True.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc13:
// CHECK:STDOUT:   %False.ref.loc13: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc13(%False.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc13:
// CHECK:STDOUT:   %.loc13_13.3: type = block_arg !if.expr.result.loc13 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc14: %False = name_ref false_, %false_
// CHECK:STDOUT:   %GreaterEq.ref.loc14: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %Negate.ref.loc14: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc14_34: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_34: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_34 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_34: <specific function> = specific_function %Convert.bound.loc14_34, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_34: init %i32 = call %Convert.specific_fn.loc14_34(%int_1.loc14) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_34.1: %i32 = value_of_initializer %int.convert_checked.loc14_34 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_34.2: %i32 = converted %int_1.loc14, %.loc14_34.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate.loc14: init %i32 = call %Negate.ref.loc14(%.loc14_34.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_0.loc14: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc14_35.1: %i32 = value_of_initializer %int.snegate.loc14 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc14_35.2: %i32 = converted %int.snegate.loc14, %.loc14_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc14_38: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14_38: <bound method> = bound_method %int_0.loc14, %impl.elem0.loc14_38 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_38: <specific function> = specific_function %Convert.bound.loc14_38, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc14_38: init %i32 = call %Convert.specific_fn.loc14_38(%int_0.loc14) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc14_38.1: %i32 = value_of_initializer %int.convert_checked.loc14_38 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc14_38.2: %i32 = converted %int_0.loc14, %.loc14_38.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %int.greater_eq.loc14: init bool = call %GreaterEq.ref.loc14(%.loc14_35.2, %.loc14_38.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc14_14.1: bool = value_of_initializer %int.greater_eq.loc14 [template = constants.%false]
// CHECK:STDOUT:   %.loc14_14.2: bool = converted %int.greater_eq.loc14, %.loc14_14.1 [template = constants.%false]
// CHECK:STDOUT:   if %.loc14_14.2 br !if.expr.then.loc14 else br !if.expr.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc14:
// CHECK:STDOUT:   %True.ref.loc14: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc14(%True.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc14:
// CHECK:STDOUT:   %False.ref.loc14: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc14(%False.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc14:
// CHECK:STDOUT:   %.loc14_14.3: type = block_arg !if.expr.result.loc14 [template = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc15: %True = name_ref true_, %true_
// CHECK:STDOUT:   %GreaterEq.ref.loc15: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_0.loc15: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %Negate.ref.loc15: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_36: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_36: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_36 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_36: <specific function> = specific_function %Convert.bound.loc15_36, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc15_36: init %i32 = call %Convert.specific_fn.loc15_36(%int_1.loc15) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_36.1: %i32 = value_of_initializer %int.convert_checked.loc15_36 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_36.2: %i32 = converted %int_1.loc15, %.loc15_36.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.snegate.loc15: init %i32 = call %Negate.ref.loc15(%.loc15_36.2) [template = constants.%int_-1]
// CHECK:STDOUT:   %impl.elem0.loc15_26: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15_26: <bound method> = bound_method %int_0.loc15, %impl.elem0.loc15_26 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_26: <specific function> = specific_function %Convert.bound.loc15_26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc15_26: init %i32 = call %Convert.specific_fn.loc15_26(%int_0.loc15) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_26.1: %i32 = value_of_initializer %int.convert_checked.loc15_26 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_26.2: %i32 = converted %int_0.loc15, %.loc15_26.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_37.1: %i32 = value_of_initializer %int.snegate.loc15 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_37.2: %i32 = converted %int.snegate.loc15, %.loc15_37.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.greater_eq.loc15: init bool = call %GreaterEq.ref.loc15(%.loc15_26.2, %.loc15_37.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc15_13.1: bool = value_of_initializer %int.greater_eq.loc15 [template = constants.%true]
// CHECK:STDOUT:   %.loc15_13.2: bool = converted %int.greater_eq.loc15, %.loc15_13.1 [template = constants.%true]
// CHECK:STDOUT:   if %.loc15_13.2 br !if.expr.then.loc15 else br !if.expr.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc15:
// CHECK:STDOUT:   %True.ref.loc15: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc15(%True.ref.loc15)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc15:
// CHECK:STDOUT:   %False.ref.loc15: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc15(%False.ref.loc15)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc15:
// CHECK:STDOUT:   %.loc15_13.3: type = block_arg !if.expr.result.loc15 [template = constants.%True]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %GreaterEq.ref: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.greater_eq: init bool = call %GreaterEq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc19_25.1: bool = value_of_initializer %int.greater_eq
// CHECK:STDOUT:   %.loc19_25.2: bool = converted %int.greater_eq, %.loc19_25.1
// CHECK:STDOUT:   return %.loc19_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %GreaterEq.type: type = fn_type @GreaterEq [template]
// CHECK:STDOUT:   %GreaterEq: %GreaterEq.type = struct_value () [template]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%true) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%true) [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%false) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%false) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     .Negate = %import_ref.3
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .GreaterEq = %GreaterEq.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %GreaterEq.decl: %GreaterEq.type = fn_decl @GreaterEq [template = constants.%GreaterEq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_61.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_61.2: type = converted %bool.make_type, %.loc4_61.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_33.1: type = splice_block %.loc4_33.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_17: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_21: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_33: init type = call %IntLiteral.ref.loc4_21() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.2: type = value_of_initializer %int_literal.make_type.loc4_33 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.3: type = converted %int_literal.make_type.loc4_33, %.loc4_33.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_55.1: type = splice_block %.loc4_55.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref.loc4_39: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref.loc4_43: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type.loc4_55: init type = call %IntLiteral.ref.loc4_43() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_55.2: type = value_of_initializer %int_literal.make_type.loc4_55 [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_55.3: type = converted %int_literal.make_type.loc4_55, %.loc4_55.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %B.patt.loc6_14.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc6_14.1, runtime_param<invalid> [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc6_18.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc6_18.3: type = converted %bool.make_type, %.loc6_18.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_14.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %B.patt.loc7_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc7_9.1, runtime_param<invalid> [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %B.ref: bool = name_ref B, %B.loc7_9.1 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %Expect.loc7_30.1: type = class_type @Expect, @Expect(constants.%B) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_13.1: type = splice_block %.loc7_13.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc7_13.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc7_13.3: type = converted %bool.make_type, %.loc7_13.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc7_9.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_30.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%B.loc6_14.1: bool) {
// CHECK:STDOUT:   %B.loc6_14.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_14.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterEq(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> bool = "int.greater_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%B.loc7_9.1: bool) {
// CHECK:STDOUT:   %B.loc7_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc7_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   %Expect.loc7_30.2: type = class_type @Expect, @Expect(%B.loc7_9.2) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_30.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_30.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%B.param_patt: bool) -> %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_42.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_42.2: init @Test.%Expect.loc7_30.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_43: init @Test.%Expect.loc7_30.2 (%Expect.1) = converted %.loc7_42.1, %.loc7_42.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_43 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc10: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_5.loc10_18: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_5.loc10_21: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int.greater_eq.loc10: init bool = call %GreaterEq.ref.loc10(%int_5.loc10_18, %int_5.loc10_21) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_23.1: bool = value_of_initializer %int.greater_eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_23.2: bool = converted %int.greater_eq.loc10, %.loc10_23.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_23.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc10: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_23.4: ref %Expect.2 = temporary %.loc10_23.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc11: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_5.loc11: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_6.loc11: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %int.greater_eq.loc11: init bool = call %GreaterEq.ref.loc11(%int_5.loc11, %int_6.loc11) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_23.1: bool = value_of_initializer %int.greater_eq.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_23.2: bool = converted %int.greater_eq.loc11, %.loc11_23.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_23.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_23.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_23.4: ref %Expect.3 = temporary %.loc11_23.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc12: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_6.loc12: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %int_5.loc12: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int.greater_eq.loc12: init bool = call %GreaterEq.ref.loc12(%int_6.loc12, %int_5.loc12) [template = constants.%true]
// CHECK:STDOUT:   %.loc12_23.1: bool = value_of_initializer %int.greater_eq.loc12 [template = constants.%true]
// CHECK:STDOUT:   %.loc12_23.2: bool = converted %int.greater_eq.loc12, %.loc12_23.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_23.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc12: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_23.4: ref %Expect.2 = temporary %.loc12_23.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc13: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc13_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc13_18: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc13_18: <bound method> = bound_method %int_1.loc13_19, %impl.elem0.loc13_18 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_18: init Core.IntLiteral = call %Op.bound.loc13_18(%int_1.loc13_19) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc13_23: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc13_22: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc13_22: <bound method> = bound_method %int_1.loc13_23, %impl.elem0.loc13_22 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_22: init Core.IntLiteral = call %Op.bound.loc13_22(%int_1.loc13_23) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_18 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_18.2: Core.IntLiteral = converted %int.snegate.loc13_18, %.loc13_18.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_22.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_22 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc13_22.2: Core.IntLiteral = converted %int.snegate.loc13_22, %.loc13_22.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.greater_eq.loc13: init bool = call %GreaterEq.ref.loc13(%.loc13_18.2, %.loc13_22.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc13_25.1: bool = value_of_initializer %int.greater_eq.loc13 [template = constants.%true]
// CHECK:STDOUT:   %.loc13_25.2: bool = converted %int.greater_eq.loc13, %.loc13_25.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_25.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc13: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_25.4: ref %Expect.2 = temporary %.loc13_25.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc14: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc14_19: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc14: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc14: <bound method> = bound_method %int_1.loc14_19, %impl.elem0.loc14 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc14: init Core.IntLiteral = call %Op.bound.loc14(%int_1.loc14_19) [template = constants.%int_-1]
// CHECK:STDOUT:   %int_1.loc14_22: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc14_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc14_18.2: Core.IntLiteral = converted %int.snegate.loc14, %.loc14_18.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.greater_eq.loc14: init bool = call %GreaterEq.ref.loc14(%.loc14_18.2, %int_1.loc14_22) [template = constants.%false]
// CHECK:STDOUT:   %.loc14_24.1: bool = value_of_initializer %int.greater_eq.loc14 [template = constants.%false]
// CHECK:STDOUT:   %.loc14_24.2: bool = converted %int.greater_eq.loc14, %.loc14_24.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc14_24.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.3 = call %Test.specific_fn.loc14() to %.loc14_24.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc14_24.4: ref %Expect.3 = temporary %.loc14_24.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc15: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc15_18: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_1.loc15_22: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc15: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15: <bound method> = bound_method %int_1.loc15_22, %impl.elem0.loc15 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc15: init Core.IntLiteral = call %Op.bound.loc15(%int_1.loc15_22) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_21.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc15_21.2: Core.IntLiteral = converted %int.snegate.loc15, %.loc15_21.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.greater_eq.loc15: init bool = call %GreaterEq.ref.loc15(%int_1.loc15_18, %.loc15_21.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc15_24.1: bool = value_of_initializer %int.greater_eq.loc15 [template = constants.%true]
// CHECK:STDOUT:   %.loc15_24.2: bool = converted %int.greater_eq.loc15, %.loc15_24.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc15_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.2 = call %Test.specific_fn.loc15() to %.loc15_24.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc15: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc15_24.4: ref %Expect.2 = temporary %.loc15_24.3, %Test.call.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%B) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%B.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%B) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%true) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%true) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%true
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%false) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%false) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%false
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- mixed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %GreaterEq.type: type = fn_type @GreaterEq [template]
// CHECK:STDOUT:   %GreaterEq: %GreaterEq.type = struct_value () [template]
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %B.patt: bool = symbolic_binding_pattern B, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%B) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete.2: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.19: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_5.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%true) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%true) [template]
// CHECK:STDOUT:   %int_6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_6.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%false) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%false) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.20: <witness> = interface_witness (%Op.14) [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.1, %Op.14 [template]
// CHECK:STDOUT:   %int_-1.1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_-1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_-1.2: %i32 = int_value -1 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .Bool = %import_ref.6
// CHECK:STDOUT:     .ImplicitAs = %import_ref.7
// CHECK:STDOUT:     .Negate = %import_ref.231
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .GreaterEq = %GreaterEq.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %GreaterEq.decl: %GreaterEq.type = fn_decl @GreaterEq [template = constants.%GreaterEq] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_47.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_47.2: type = converted %bool.make_type, %.loc4_47.1 [template = bool]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_33.1: type = splice_block %.loc4_33.3 [template = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.2: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:       %.loc4_33.3: type = converted %int_literal.make_type, %.loc4_33.2 [template = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %.loc4_39: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param2
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %B.patt.loc6_14.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc6_14.1, runtime_param<invalid> [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc6_18.1: type = splice_block %.loc6_18.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc6_18.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc6_18.3: type = converted %bool.make_type, %.loc6_18.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc6_14.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %B.patt.loc7_9.1: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %B.param_patt: bool = value_param_pattern %B.patt.loc7_9.1, runtime_param<invalid> [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %B.ref: bool = name_ref B, %B.loc7_9.1 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %Expect.loc7_30.1: type = class_type @Expect, @Expect(constants.%B) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %B.param: bool = value_param runtime_param<invalid>
// CHECK:STDOUT:     %.loc7_13.1: type = splice_block %.loc7_13.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc7_13.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc7_13.3: type = converted %bool.make_type, %.loc7_13.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %B.loc7_9.1: bool = bind_symbolic_name B, 0, %B.param [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_30.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_30.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%B.loc6_14.1: bool) {
// CHECK:STDOUT:   %B.loc6_14.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc6_14.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc6_14.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc6_14.2 (constants.%B.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Expect.1
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GreaterEq(%a.param_patt: Core.IntLiteral, %b.param_patt: %i32) -> bool = "int.greater_eq";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%B.loc7_9.1: bool) {
// CHECK:STDOUT:   %B.loc7_9.2: bool = bind_symbolic_name B, 0 [symbolic = %B.loc7_9.2 (constants.%B)]
// CHECK:STDOUT:   %B.patt.loc7_9.2: bool = symbolic_binding_pattern B, 0 [symbolic = %B.patt.loc7_9.2 (constants.%B.patt)]
// CHECK:STDOUT:   %Expect.loc7_30.2: type = class_type @Expect, @Expect(%B.loc7_9.2) [symbolic = %Expect.loc7_30.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_30.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete.2)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_30.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%B.param_patt: bool) -> %return.param_patt: @Test.%Expect.loc7_30.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_42.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_42.2: init @Test.%Expect.loc7_30.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_43: init @Test.%Expect.loc7_30.2 (%Expect.1) = converted %.loc7_42.1, %.loc7_42.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_43 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc10: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_5.loc10_18: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_5.loc10_21: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc10: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc10: <bound method> = bound_method %int_5.loc10_21, %impl.elem0.loc10 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10: <specific function> = specific_function %Convert.bound.loc10, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %i32 = call %Convert.specific_fn.loc10(%int_5.loc10_21) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_21.1: %i32 = value_of_initializer %int.convert_checked.loc10 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc10_21.2: %i32 = converted %int_5.loc10_21, %.loc10_21.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %int.greater_eq.loc10: init bool = call %GreaterEq.ref.loc10(%int_5.loc10_18, %.loc10_21.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc10_23.1: bool = value_of_initializer %int.greater_eq.loc10 [template = constants.%true]
// CHECK:STDOUT:   %.loc10_23.2: bool = converted %int.greater_eq.loc10, %.loc10_23.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_23.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc10: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_23.4: ref %Expect.2 = temporary %.loc10_23.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc11: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_5.loc11: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_6.loc11: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_6.loc11, %impl.elem0.loc11 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_6.loc11) [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %int.convert_checked.loc11 [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_6.loc11, %.loc11_21.1 [template = constants.%int_6.2]
// CHECK:STDOUT:   %int.greater_eq.loc11: init bool = call %GreaterEq.ref.loc11(%int_5.loc11, %.loc11_21.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc11_23.1: bool = value_of_initializer %int.greater_eq.loc11 [template = constants.%false]
// CHECK:STDOUT:   %.loc11_23.2: bool = converted %int.greater_eq.loc11, %.loc11_23.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_23.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_23.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc11: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_23.4: ref %Expect.3 = temporary %.loc11_23.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc12: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_6.loc12: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %int_5.loc12: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_5.loc12, %impl.elem0.loc12 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.specific_fn.loc12(%int_5.loc12) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc12_21.1: %i32 = value_of_initializer %int.convert_checked.loc12 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc12_21.2: %i32 = converted %int_5.loc12, %.loc12_21.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %int.greater_eq.loc12: init bool = call %GreaterEq.ref.loc12(%int_6.loc12, %.loc12_21.2) [template = constants.%true]
// CHECK:STDOUT:   %.loc12_23.1: bool = value_of_initializer %int.greater_eq.loc12 [template = constants.%true]
// CHECK:STDOUT:   %.loc12_23.2: bool = converted %int.greater_eq.loc12, %.loc12_23.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc12_23.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.2 = call %Test.specific_fn.loc12() to %.loc12_23.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc12: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc12_23.4: ref %Expect.2 = temporary %.loc12_23.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc13: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc13_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_18: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc13_18: <bound method> = bound_method %int_1.loc13_19, %impl.elem0.loc13_18 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_18: init Core.IntLiteral = call %Op.bound.loc13_18(%int_1.loc13_19) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc13_23: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_22.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc13_22: <bound method> = bound_method %int_1.loc13_23, %impl.elem0.loc13_22.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc13_22: init Core.IntLiteral = call %Op.bound.loc13_22(%int_1.loc13_23) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc13_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_18 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc13_18.2: Core.IntLiteral = converted %int.snegate.loc13_18, %.loc13_18.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc13_22.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc13: <bound method> = bound_method %int.snegate.loc13_22, %impl.elem0.loc13_22.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13: <specific function> = specific_function %Convert.bound.loc13, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc13_22.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13_22 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc13_22.2: Core.IntLiteral = converted %int.snegate.loc13_22, %.loc13_22.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.specific_fn.loc13(%.loc13_22.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc13_22.3: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc13_22.4: %i32 = converted %int.snegate.loc13_22, %.loc13_22.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.greater_eq.loc13: init bool = call %GreaterEq.ref.loc13(%.loc13_18.2, %.loc13_22.4) [template = constants.%true]
// CHECK:STDOUT:   %.loc13_25.1: bool = value_of_initializer %int.greater_eq.loc13 [template = constants.%true]
// CHECK:STDOUT:   %.loc13_25.2: bool = converted %int.greater_eq.loc13, %.loc13_25.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc13_25.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.2 = call %Test.specific_fn.loc13() to %.loc13_25.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc13: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc13_25.4: ref %Expect.2 = temporary %.loc13_25.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc14: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc14_19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc14_18: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc14: <bound method> = bound_method %int_1.loc14_19, %impl.elem0.loc14_18 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc14: init Core.IntLiteral = call %Op.bound.loc14(%int_1.loc14_19) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int_1.loc14_22: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc14_18.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc14_18.2: Core.IntLiteral = converted %int.snegate.loc14, %.loc14_18.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc14_22: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc14: <bound method> = bound_method %int_1.loc14_22, %impl.elem0.loc14_22 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc14: <specific function> = specific_function %Convert.bound.loc14, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %Convert.specific_fn.loc14(%int_1.loc14_22) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_22.1: %i32 = value_of_initializer %int.convert_checked.loc14 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_22.2: %i32 = converted %int_1.loc14_22, %.loc14_22.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int.greater_eq.loc14: init bool = call %GreaterEq.ref.loc14(%.loc14_18.2, %.loc14_22.2) [template = constants.%false]
// CHECK:STDOUT:   %.loc14_24.1: bool = value_of_initializer %int.greater_eq.loc14 [template = constants.%false]
// CHECK:STDOUT:   %.loc14_24.2: bool = converted %int.greater_eq.loc14, %.loc14_24.1 [template = constants.%false]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%false) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc14_24.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.3 = call %Test.specific_fn.loc14() to %.loc14_24.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%false) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc14_24.4: ref %Expect.3 = temporary %.loc14_24.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %GreaterEq.ref.loc15: %GreaterEq.type = name_ref GreaterEq, file.%GreaterEq.decl [template = constants.%GreaterEq]
// CHECK:STDOUT:   %int_1.loc15_18: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_1.loc15_22: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_21.1: %Op.type.13 = interface_witness_access constants.%interface.20, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound.loc15: <bound method> = bound_method %int_1.loc15_22, %impl.elem0.loc15_21.1 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate.loc15: init Core.IntLiteral = call %Op.bound.loc15(%int_1.loc15_22) [template = constants.%int_-1.1]
// CHECK:STDOUT:   %impl.elem0.loc15_21.2: %Convert.type.2 = interface_witness_access constants.%interface.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int.snegate.loc15, %impl.elem0.loc15_21.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc15_21.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %.loc15_21.2: Core.IntLiteral = converted %int.snegate.loc15, %.loc15_21.1 [template = constants.%int_-1.1]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.specific_fn.loc15(%.loc15_21.2) [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_21.3: %i32 = value_of_initializer %int.convert_checked.loc15 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %.loc15_21.4: %i32 = converted %int.snegate.loc15, %.loc15_21.3 [template = constants.%int_-1.2]
// CHECK:STDOUT:   %int.greater_eq.loc15: init bool = call %GreaterEq.ref.loc15(%int_1.loc15_18, %.loc15_21.4) [template = constants.%true]
// CHECK:STDOUT:   %.loc15_24.1: bool = value_of_initializer %int.greater_eq.loc15 [template = constants.%true]
// CHECK:STDOUT:   %.loc15_24.2: bool = converted %int.greater_eq.loc15, %.loc15_24.1 [template = constants.%true]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%true) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc15_24.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.2 = call %Test.specific_fn.loc15() to %.loc15_24.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %true.loc15: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%true) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc15_24.4: ref %Expect.2 = temporary %.loc15_24.3, %Test.call.loc15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%B) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%B.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%B) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%B
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%true) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%true
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%true) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%true
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%true
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%false) {
// CHECK:STDOUT:   %B.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc7_9.2 => constants.%false
// CHECK:STDOUT:   %Expect.loc7_30.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%false) {
// CHECK:STDOUT:   %B.loc6_14.2 => constants.%false
// CHECK:STDOUT:   %B.patt.loc6_14.2 => constants.%false
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
