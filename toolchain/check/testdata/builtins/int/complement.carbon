// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/complement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/complement.carbon

// --- int_complement.carbon

library "[[@TEST_NAME]]";

fn Complement(a: i32) -> i32 = "int.complement";
fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(Complement(0x123456), 0xFFFFFF)];
let arr_p: [i32; 0xEDCBA9]* = &arr;

fn RuntimeCall(a: i32) -> i32 {
  return Complement(a);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn Complement(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.complement";

class Expect(N:! Core.IntLiteral()) {}
fn Test(N:! Core.IntLiteral()) -> Expect(N) { return {}; }

fn F() {
  Test(Complement(0)) as Expect(-1);
  Test(Complement(1)) as Expect(-2);
  Test(Complement(-1)) as Expect(0);
  Test(Complement(-0x7FFF_FFFF_FFFF_FFFF)) as Expect(0x7FFF_FFFF_FFFF_FFFE);
  Test(Complement(-0x8000_0000_0000_0000)) as Expect(0x7FFF_FFFF_FFFF_FFFF);
  Test(Complement(0x7FFF_FFFF_FFFF_FFFF)) as Expect(-0x8000_0000_0000_0000);
  Test(Complement(0x8000_0000_0000_0000)) as Expect(-0x8000_0000_0000_0001);
}

// --- fail_literal_runtime.carbon

library "[[@TEST_NAME]]";

fn Complement(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.complement";

fn F(a: Core.IntLiteral()) -> Core.IntLiteral() {
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE+6]]:10: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   return Complement(a);
  // CHECK:STDERR:          ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE-6]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn Complement(a: Core.IntLiteral()) -> Core.IntLiteral() = "int.complement";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return Complement(a);
}

// CHECK:STDOUT: --- int_complement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Complement.type: type = fn_type @Complement [template]
// CHECK:STDOUT:   %Complement: %Complement.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %int_1193046.1: Core.IntLiteral = int_value 1193046 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1193046.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1193046.2: %i32 = int_value 1193046 [template]
// CHECK:STDOUT:   %int_-1193047: %i32 = int_value -1193047 [template]
// CHECK:STDOUT:   %int_16777215.1: Core.IntLiteral = int_value 16777215 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_16777215.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_16777215.2: %i32 = int_value 16777215 [template]
// CHECK:STDOUT:   %int_15584169.1: %i32 = int_value 15584169 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_15584169.1, %Convert.11 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %int_15584169.2: Core.IntLiteral = int_value 15584169 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_15584169.2, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Complement = %Complement.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Complement.decl: %Complement.type = fn_decl @Complement [template = constants.%Complement] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_26: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, %And.decl [template = constants.%And]
// CHECK:STDOUT:   %Complement.ref: %Complement.type = name_ref Complement, %Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_1193046: Core.IntLiteral = int_value 1193046 [template = constants.%int_1193046.1]
// CHECK:STDOUT:   %impl.elem0.loc7_31: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc7_31: <bound method> = bound_method %int_1193046, %impl.elem0.loc7_31 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_31: <specific function> = specific_function %Convert.bound.loc7_31, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc7_31: init %i32 = call %Convert.specific_fn.loc7_31(%int_1193046) [template = constants.%int_1193046.2]
// CHECK:STDOUT:   %.loc7_31.1: %i32 = value_of_initializer %int.convert_checked.loc7_31 [template = constants.%int_1193046.2]
// CHECK:STDOUT:   %.loc7_31.2: %i32 = converted %int_1193046, %.loc7_31.1 [template = constants.%int_1193046.2]
// CHECK:STDOUT:   %int.complement: init %i32 = call %Complement.ref(%.loc7_31.2) [template = constants.%int_-1193047]
// CHECK:STDOUT:   %int_16777215: Core.IntLiteral = int_value 16777215 [template = constants.%int_16777215.1]
// CHECK:STDOUT:   %.loc7_39.1: %i32 = value_of_initializer %int.complement [template = constants.%int_-1193047]
// CHECK:STDOUT:   %.loc7_39.2: %i32 = converted %int.complement, %.loc7_39.1 [template = constants.%int_-1193047]
// CHECK:STDOUT:   %impl.elem0.loc7_42: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc7_42: <bound method> = bound_method %int_16777215, %impl.elem0.loc7_42 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_42: <specific function> = specific_function %Convert.bound.loc7_42, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc7_42: init %i32 = call %Convert.specific_fn.loc7_42(%int_16777215) [template = constants.%int_16777215.2]
// CHECK:STDOUT:   %.loc7_42.1: %i32 = value_of_initializer %int.convert_checked.loc7_42 [template = constants.%int_16777215.2]
// CHECK:STDOUT:   %.loc7_42.2: %i32 = converted %int_16777215, %.loc7_42.1 [template = constants.%int_16777215.2]
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%.loc7_39.2, %.loc7_42.2) [template = constants.%int_15584169.1]
// CHECK:STDOUT:   %impl.elem0.loc7_50: %Convert.type.5 = interface_witness_access constants.%interface.6, element0 [template = constants.%Convert.11]
// CHECK:STDOUT:   %Convert.bound.loc7_50: <bound method> = bound_method %int.and, %impl.elem0.loc7_50 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_50: <specific function> = specific_function %Convert.bound.loc7_50, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc7_50.1: %i32 = value_of_initializer %int.and [template = constants.%int_15584169.1]
// CHECK:STDOUT:   %.loc7_50.2: %i32 = converted %int.and, %.loc7_50.1 [template = constants.%int_15584169.1]
// CHECK:STDOUT:   %int.convert_checked.loc7_50: init Core.IntLiteral = call %Convert.specific_fn.loc7_50(%.loc7_50.2) [template = constants.%int_15584169.2]
// CHECK:STDOUT:   %.loc7_50.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc7_50 [template = constants.%int_15584169.2]
// CHECK:STDOUT:   %.loc7_50.4: Core.IntLiteral = converted %int.and, %.loc7_50.3 [template = constants.%int_15584169.2]
// CHECK:STDOUT:   %array_type.loc7: type = array_type %.loc7_50.4, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_15584169: Core.IntLiteral = int_value 15584169 [template = constants.%int_15584169.2]
// CHECK:STDOUT:   %array_type.loc8: type = array_type %int_15584169, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Complement(%a.param_patt: %i32) -> %i32 = "int.complement";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Complement.ref: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int.complement: init %i32 = call %Complement.ref(%a.ref)
// CHECK:STDOUT:   %.loc11_23.1: %i32 = value_of_initializer %int.complement
// CHECK:STDOUT:   %.loc11_23.2: %i32 = converted %int.complement, %.loc11_23.1
// CHECK:STDOUT:   return %.loc11_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %array_type = name_ref arr, file.%arr
// CHECK:STDOUT:   %addr: %ptr = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, %addr
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Complement.type: type = fn_type @Complement [template]
// CHECK:STDOUT:   %Complement: %Complement.type = struct_value () [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %Expect.type: type = generic_class_type @Expect [template]
// CHECK:STDOUT:   %Expect.generic: %Expect.type = struct_value () [template]
// CHECK:STDOUT:   %Expect.1: type = class_type @Expect, @Expect(%N) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Expect.1 [symbolic]
// CHECK:STDOUT:   %Expect.val.1: %Expect.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %Expect.2: type = class_type @Expect, @Expect(%int_-1) [template]
// CHECK:STDOUT:   %Test.specific_fn.1: <specific function> = specific_function %Test, @Test(%int_-1) [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-2: Core.IntLiteral = int_value -2 [template]
// CHECK:STDOUT:   %Expect.3: type = class_type @Expect, @Expect(%int_-2) [template]
// CHECK:STDOUT:   %Test.specific_fn.2: <specific function> = specific_function %Test, @Test(%int_-2) [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_2, %Op.2 [template]
// CHECK:STDOUT:   %Expect.4: type = class_type @Expect, @Expect(%int_0) [template]
// CHECK:STDOUT:   %Test.specific_fn.3: <specific function> = specific_function %Test, @Test(%int_0) [template]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [template]
// CHECK:STDOUT:   %Op.bound.3: <bound method> = bound_method %int_9223372036854775807, %Op.2 [template]
// CHECK:STDOUT:   %int_-9223372036854775807: Core.IntLiteral = int_value -9223372036854775807 [template]
// CHECK:STDOUT:   %int_9223372036854775806: Core.IntLiteral = int_value 9223372036854775806 [template]
// CHECK:STDOUT:   %Expect.5: type = class_type @Expect, @Expect(%int_9223372036854775806) [template]
// CHECK:STDOUT:   %Test.specific_fn.4: <specific function> = specific_function %Test, @Test(%int_9223372036854775806) [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %Op.bound.4: <bound method> = bound_method %int_9223372036854775808, %Op.2 [template]
// CHECK:STDOUT:   %int_-9223372036854775808: Core.IntLiteral = int_value -9223372036854775808 [template]
// CHECK:STDOUT:   %Expect.6: type = class_type @Expect, @Expect(%int_9223372036854775807) [template]
// CHECK:STDOUT:   %Test.specific_fn.5: <specific function> = specific_function %Test, @Test(%int_9223372036854775807) [template]
// CHECK:STDOUT:   %Expect.7: type = class_type @Expect, @Expect(%int_-9223372036854775808) [template]
// CHECK:STDOUT:   %Test.specific_fn.6: <specific function> = specific_function %Test, @Test(%int_-9223372036854775808) [template]
// CHECK:STDOUT:   %int_-9223372036854775809: Core.IntLiteral = int_value -9223372036854775809 [template]
// CHECK:STDOUT:   %Expect.8: type = class_type @Expect, @Expect(%int_-9223372036854775809) [template]
// CHECK:STDOUT:   %Test.specific_fn.7: <specific function> = specific_function %Test, @Test(%int_-9223372036854775809) [template]
// CHECK:STDOUT:   %int_9223372036854775809: Core.IntLiteral = int_value 9223372036854775809 [template]
// CHECK:STDOUT:   %Op.bound.5: <bound method> = bound_method %int_9223372036854775809, %Op.2 [template]
// CHECK:STDOUT:   %Expect.val.2: %Expect.2 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.3: %Expect.3 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.4: %Expect.4 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.5: %Expect.5 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.6: %Expect.6 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.7: %Expect.7 = struct_value () [template]
// CHECK:STDOUT:   %Expect.val.8: %Expect.8 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Negate = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Complement = %Complement.decl
// CHECK:STDOUT:     .Expect = %Expect.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Complement.decl: %Complement.type = fn_decl @Complement [template = constants.%Complement] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_18: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_22: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_34: init type = call %IntLiteral.ref.loc4_22() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int_literal.make_type.loc4_34 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int_literal.make_type.loc4_34, %.loc4_34.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_40: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_44: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_56: init type = call %IntLiteral.ref.loc4_44() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_56.1: type = value_of_initializer %int_literal.make_type.loc4_56 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_56.2: type = converted %int_literal.make_type.loc4_56, %.loc4_56.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Expect.decl: %Expect.type = class_decl @Expect [template = constants.%Expect.generic] {
// CHECK:STDOUT:     %N.patt.loc6_14.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc6_14.1, runtime_param<invalid> [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_34.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_34.2: type = converted %int_literal.make_type, %.loc6_34.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_14.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {
// CHECK:STDOUT:     %N.patt.loc7_9.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %N.param_patt: Core.IntLiteral = value_param_pattern %N.patt.loc7_9.1, runtime_param<invalid> [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:     %return.patt: @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type: init type = call %IntLiteral.ref() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc7_29.1: type = value_of_initializer %int_literal.make_type [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc7_29.2: type = converted %int_literal.make_type, %.loc7_29.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Expect.ref: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:     %N.ref: Core.IntLiteral = name_ref N, %N.loc7_9.1 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %Expect.loc7_43.1: type = class_type @Expect, @Expect(constants.%N) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc7_9.1: Core.IntLiteral = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:     %return.param: ref @Test.%Expect.loc7_43.2 (%Expect.1) = out_param runtime_param0
// CHECK:STDOUT:     %return: ref @Test.%Expect.loc7_43.2 (%Expect.1) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Expect(%N.loc6_14.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_14.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_14.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_14.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_14.2 (constants.%N.patt)]
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
// CHECK:STDOUT: fn @Complement(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.complement";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Test(%N.loc7_9.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_9.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_9.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc7_9.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_9.2 (constants.%N.patt)]
// CHECK:STDOUT:   %Expect.loc7_43.2: type = class_type @Expect, @Expect(%N.loc7_9.2) [symbolic = %Expect.loc7_43.2 (constants.%Expect.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Test.%Expect.loc7_43.2 (%Expect.1) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %Expect.val: @Test.%Expect.loc7_43.2 (%Expect.1) = struct_value () [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: Core.IntLiteral) -> %return.param_patt: @Test.%Expect.loc7_43.2 (%Expect.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc7_55.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc7_55.2: init @Test.%Expect.loc7_43.2 (%Expect.1) = class_init (), %return [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     %.loc7_56: init @Test.%Expect.loc7_43.2 (%Expect.1) = converted %.loc7_55.1, %.loc7_55.2 [symbolic = %Expect.val (constants.%Expect.val.1)]
// CHECK:STDOUT:     return %.loc7_56 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Test.ref.loc10: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Complement.ref.loc10: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_0.loc10: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int.complement.loc10: init Core.IntLiteral = call %Complement.ref.loc10(%int_0.loc10) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_21.1: Core.IntLiteral = value_of_initializer %int.complement.loc10 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_21.2: Core.IntLiteral = converted %int.complement.loc10, %.loc10_21.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Test.specific_fn.loc10: <specific function> = specific_function %Test.ref.loc10, @Test(constants.%int_-1) [template = constants.%Test.specific_fn.1]
// CHECK:STDOUT:   %.loc10_21.3: ref %Expect.2 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc10: init %Expect.2 = call %Test.specific_fn.loc10() to %.loc10_21.3
// CHECK:STDOUT:   %Expect.ref.loc10: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc10: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc10: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc10: init Core.IntLiteral = call %Op.bound.loc10(%int_1.loc10) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_35.1: Core.IntLiteral = value_of_initializer %int.snegate.loc10 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_35.2: Core.IntLiteral = converted %int.snegate.loc10, %.loc10_35.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %Expect.loc10: type = class_type @Expect, @Expect(constants.%int_-1) [template = constants.%Expect.2]
// CHECK:STDOUT:   %.loc10_21.4: ref %Expect.2 = temporary %.loc10_21.3, %Test.call.loc10
// CHECK:STDOUT:   %Test.ref.loc11: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Complement.ref.loc11: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int.complement.loc11: init Core.IntLiteral = call %Complement.ref.loc11(%int_1.loc11) [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc11_21.1: Core.IntLiteral = value_of_initializer %int.complement.loc11 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc11_21.2: Core.IntLiteral = converted %int.complement.loc11, %.loc11_21.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %Test.specific_fn.loc11: <specific function> = specific_function %Test.ref.loc11, @Test(constants.%int_-2) [template = constants.%Test.specific_fn.2]
// CHECK:STDOUT:   %.loc11_21.3: ref %Expect.3 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc11: init %Expect.3 = call %Test.specific_fn.loc11() to %.loc11_21.3
// CHECK:STDOUT:   %Expect.ref.loc11: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc11: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc11: <bound method> = bound_method %int_2, %impl.elem0.loc11 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %int.snegate.loc11: init Core.IntLiteral = call %Op.bound.loc11(%int_2) [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc11_35.1: Core.IntLiteral = value_of_initializer %int.snegate.loc11 [template = constants.%int_-2]
// CHECK:STDOUT:   %.loc11_35.2: Core.IntLiteral = converted %int.snegate.loc11, %.loc11_35.1 [template = constants.%int_-2]
// CHECK:STDOUT:   %Expect.loc11: type = class_type @Expect, @Expect(constants.%int_-2) [template = constants.%Expect.3]
// CHECK:STDOUT:   %.loc11_21.4: ref %Expect.3 = temporary %.loc11_21.3, %Test.call.loc11
// CHECK:STDOUT:   %Test.ref.loc12: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Complement.ref.loc12: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc12: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc12: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate.loc12: init Core.IntLiteral = call %Op.bound.loc12(%int_1.loc12) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc12_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc12 [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc12_19.2: Core.IntLiteral = converted %int.snegate.loc12, %.loc12_19.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.complement.loc12: init Core.IntLiteral = call %Complement.ref.loc12(%.loc12_19.2) [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_22.1: Core.IntLiteral = value_of_initializer %int.complement.loc12 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_22.2: Core.IntLiteral = converted %int.complement.loc12, %.loc12_22.1 [template = constants.%int_0]
// CHECK:STDOUT:   %Test.specific_fn.loc12: <specific function> = specific_function %Test.ref.loc12, @Test(constants.%int_0) [template = constants.%Test.specific_fn.3]
// CHECK:STDOUT:   %.loc12_22.3: ref %Expect.4 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc12: init %Expect.4 = call %Test.specific_fn.loc12() to %.loc12_22.3
// CHECK:STDOUT:   %Expect.ref.loc12: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_0.loc12: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %Expect.loc12: type = class_type @Expect, @Expect(constants.%int_0) [template = constants.%Expect.4]
// CHECK:STDOUT:   %.loc12_22.4: ref %Expect.4 = temporary %.loc12_22.3, %Test.call.loc12
// CHECK:STDOUT:   %Test.ref.loc13: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Complement.ref.loc13: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_9223372036854775807.loc13: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %impl.elem0.loc13: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc13: <bound method> = bound_method %int_9223372036854775807.loc13, %impl.elem0.loc13 [template = constants.%Op.bound.3]
// CHECK:STDOUT:   %int.snegate.loc13: init Core.IntLiteral = call %Op.bound.loc13(%int_9223372036854775807.loc13) [template = constants.%int_-9223372036854775807]
// CHECK:STDOUT:   %.loc13_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc13 [template = constants.%int_-9223372036854775807]
// CHECK:STDOUT:   %.loc13_19.2: Core.IntLiteral = converted %int.snegate.loc13, %.loc13_19.1 [template = constants.%int_-9223372036854775807]
// CHECK:STDOUT:   %int.complement.loc13: init Core.IntLiteral = call %Complement.ref.loc13(%.loc13_19.2) [template = constants.%int_9223372036854775806]
// CHECK:STDOUT:   %.loc13_42.1: Core.IntLiteral = value_of_initializer %int.complement.loc13 [template = constants.%int_9223372036854775806]
// CHECK:STDOUT:   %.loc13_42.2: Core.IntLiteral = converted %int.complement.loc13, %.loc13_42.1 [template = constants.%int_9223372036854775806]
// CHECK:STDOUT:   %Test.specific_fn.loc13: <specific function> = specific_function %Test.ref.loc13, @Test(constants.%int_9223372036854775806) [template = constants.%Test.specific_fn.4]
// CHECK:STDOUT:   %.loc13_42.3: ref %Expect.5 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc13: init %Expect.5 = call %Test.specific_fn.loc13() to %.loc13_42.3
// CHECK:STDOUT:   %Expect.ref.loc13: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775806: Core.IntLiteral = int_value 9223372036854775806 [template = constants.%int_9223372036854775806]
// CHECK:STDOUT:   %Expect.loc13: type = class_type @Expect, @Expect(constants.%int_9223372036854775806) [template = constants.%Expect.5]
// CHECK:STDOUT:   %.loc13_42.4: ref %Expect.5 = temporary %.loc13_42.3, %Test.call.loc13
// CHECK:STDOUT:   %Test.ref.loc14: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Complement.ref.loc14: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_9223372036854775808.loc14: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc14: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc14: <bound method> = bound_method %int_9223372036854775808.loc14, %impl.elem0.loc14 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc14: init Core.IntLiteral = call %Op.bound.loc14(%int_9223372036854775808.loc14) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc14_19.1: Core.IntLiteral = value_of_initializer %int.snegate.loc14 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc14_19.2: Core.IntLiteral = converted %int.snegate.loc14, %.loc14_19.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %int.complement.loc14: init Core.IntLiteral = call %Complement.ref.loc14(%.loc14_19.2) [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %.loc14_42.1: Core.IntLiteral = value_of_initializer %int.complement.loc14 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %.loc14_42.2: Core.IntLiteral = converted %int.complement.loc14, %.loc14_42.1 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %Test.specific_fn.loc14: <specific function> = specific_function %Test.ref.loc14, @Test(constants.%int_9223372036854775807) [template = constants.%Test.specific_fn.5]
// CHECK:STDOUT:   %.loc14_42.3: ref %Expect.6 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc14: init %Expect.6 = call %Test.specific_fn.loc14() to %.loc14_42.3
// CHECK:STDOUT:   %Expect.ref.loc14: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775807.loc14: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %Expect.loc14: type = class_type @Expect, @Expect(constants.%int_9223372036854775807) [template = constants.%Expect.6]
// CHECK:STDOUT:   %.loc14_42.4: ref %Expect.6 = temporary %.loc14_42.3, %Test.call.loc14
// CHECK:STDOUT:   %Test.ref.loc15: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Complement.ref.loc15: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_9223372036854775807.loc15: Core.IntLiteral = int_value 9223372036854775807 [template = constants.%int_9223372036854775807]
// CHECK:STDOUT:   %int.complement.loc15: init Core.IntLiteral = call %Complement.ref.loc15(%int_9223372036854775807.loc15) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc15_41.1: Core.IntLiteral = value_of_initializer %int.complement.loc15 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc15_41.2: Core.IntLiteral = converted %int.complement.loc15, %.loc15_41.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %Test.specific_fn.loc15: <specific function> = specific_function %Test.ref.loc15, @Test(constants.%int_-9223372036854775808) [template = constants.%Test.specific_fn.6]
// CHECK:STDOUT:   %.loc15_41.3: ref %Expect.7 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc15: init %Expect.7 = call %Test.specific_fn.loc15() to %.loc15_41.3
// CHECK:STDOUT:   %Expect.ref.loc15: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775808.loc15: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %impl.elem0.loc15: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc15: <bound method> = bound_method %int_9223372036854775808.loc15, %impl.elem0.loc15 [template = constants.%Op.bound.4]
// CHECK:STDOUT:   %int.snegate.loc15: init Core.IntLiteral = call %Op.bound.loc15(%int_9223372036854775808.loc15) [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc15_75.1: Core.IntLiteral = value_of_initializer %int.snegate.loc15 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %.loc15_75.2: Core.IntLiteral = converted %int.snegate.loc15, %.loc15_75.1 [template = constants.%int_-9223372036854775808]
// CHECK:STDOUT:   %Expect.loc15: type = class_type @Expect, @Expect(constants.%int_-9223372036854775808) [template = constants.%Expect.7]
// CHECK:STDOUT:   %.loc15_41.4: ref %Expect.7 = temporary %.loc15_41.3, %Test.call.loc15
// CHECK:STDOUT:   %Test.ref.loc16: %Test.type = name_ref Test, file.%Test.decl [template = constants.%Test]
// CHECK:STDOUT:   %Complement.ref.loc16: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_9223372036854775808.loc16: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %int.complement.loc16: init Core.IntLiteral = call %Complement.ref.loc16(%int_9223372036854775808.loc16) [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc16_41.1: Core.IntLiteral = value_of_initializer %int.complement.loc16 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc16_41.2: Core.IntLiteral = converted %int.complement.loc16, %.loc16_41.1 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %Test.specific_fn.loc16: <specific function> = specific_function %Test.ref.loc16, @Test(constants.%int_-9223372036854775809) [template = constants.%Test.specific_fn.7]
// CHECK:STDOUT:   %.loc16_41.3: ref %Expect.8 = temporary_storage
// CHECK:STDOUT:   %Test.call.loc16: init %Expect.8 = call %Test.specific_fn.loc16() to %.loc16_41.3
// CHECK:STDOUT:   %Expect.ref.loc16: %Expect.type = name_ref Expect, file.%Expect.decl [template = constants.%Expect.generic]
// CHECK:STDOUT:   %int_9223372036854775809: Core.IntLiteral = int_value 9223372036854775809 [template = constants.%int_9223372036854775809]
// CHECK:STDOUT:   %impl.elem0.loc16: %Op.type.1 = interface_witness_access constants.%interface, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc16: <bound method> = bound_method %int_9223372036854775809, %impl.elem0.loc16 [template = constants.%Op.bound.5]
// CHECK:STDOUT:   %int.snegate.loc16: init Core.IntLiteral = call %Op.bound.loc16(%int_9223372036854775809) [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc16_75.1: Core.IntLiteral = value_of_initializer %int.snegate.loc16 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %.loc16_75.2: Core.IntLiteral = converted %int.snegate.loc16, %.loc16_75.1 [template = constants.%int_-9223372036854775809]
// CHECK:STDOUT:   %Expect.loc16: type = class_type @Expect, @Expect(constants.%int_-9223372036854775809) [template = constants.%Expect.8]
// CHECK:STDOUT:   %.loc16_41.4: ref %Expect.8 = temporary %.loc16_41.3, %Test.call.loc16
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%N) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(@Test.%N.loc7_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%N) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%N
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-1
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-1) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-2) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-2
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-2
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-2) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-2
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_0) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_0
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_0) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_0
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_9223372036854775806) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_9223372036854775806
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_9223372036854775806
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_9223372036854775806) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_9223372036854775806
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_9223372036854775806
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_9223372036854775807) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_9223372036854775807
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_9223372036854775807
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_9223372036854775807) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_9223372036854775807
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_9223372036854775807
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-9223372036854775808) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-9223372036854775808) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-9223372036854775808
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Test(constants.%int_-9223372036854775809) {
// CHECK:STDOUT:   %N.loc7_9.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:   %N.patt.loc7_9.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:   %Expect.loc7_43.2 => constants.%Expect.8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %Expect.val => constants.%Expect.val.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Expect(constants.%int_-9223372036854775809) {
// CHECK:STDOUT:   %N.loc6_14.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:   %N.patt.loc6_14.2 => constants.%int_-9223372036854775809
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_literal_runtime.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %Complement.type: type = fn_type @Complement [template]
// CHECK:STDOUT:   %Complement: %Complement.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Complement = %Complement.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Complement.decl: %Complement.type = fn_decl @Complement [template = constants.%Complement] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_18: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_22: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_34: init type = call %IntLiteral.ref.loc4_22() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_34.1: type = value_of_initializer %int_literal.make_type.loc4_34 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_34.2: type = converted %int_literal.make_type.loc4_34, %.loc4_34.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_40: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_44: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_56: init type = call %IntLiteral.ref.loc4_44() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_56.1: type = value_of_initializer %int_literal.make_type.loc4_56 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_56.2: type = converted %int_literal.make_type.loc4_56, %.loc4_56.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc6_9: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc6_13: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc6_25: init type = call %IntLiteral.ref.loc6_13() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_25.1: type = value_of_initializer %int_literal.make_type.loc6_25 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_25.2: type = converted %int_literal.make_type.loc6_25, %.loc6_25.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc6_31: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc6_35: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc6_47: init type = call %IntLiteral.ref.loc6_35() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_47.1: type = value_of_initializer %int_literal.make_type.loc6_47 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc6_47.2: type = converted %int_literal.make_type.loc6_47, %.loc6_47.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Complement(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.complement";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Complement.ref: %Complement.type = name_ref Complement, file.%Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %a.ref: Core.IntLiteral = name_ref a, %a
// CHECK:STDOUT:   %int.complement: init Core.IntLiteral = call %Complement.ref(%a.ref)
// CHECK:STDOUT:   %.loc13_23.1: Core.IntLiteral = value_of_initializer %int.complement
// CHECK:STDOUT:   %.loc13_23.2: Core.IntLiteral = converted %int.complement, %.loc13_23.1
// CHECK:STDOUT:   return %.loc13_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
