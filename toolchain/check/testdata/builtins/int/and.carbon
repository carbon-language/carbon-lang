// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/and.carbon

// --- int_and.carbon

library "[[@TEST_NAME]]";

fn And(a: i32, b: i32) -> i32 = "int.and";

var arr: [i32; And(12, 10)];
let arr_p: [i32; 8]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return And(a, b);
}

// --- literal.carbon

library "[[@TEST_NAME]]";

fn AndLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

var arr_lit: [i32; AndLit(12, 10)];
let arr_lit_p: [i32; 8]* = &arr_lit;

// & between a positive and negative literal should produce a positive value.
var arr_lit_large: [i32; AndLit(0x8000_0000_0000_0000, -1) >> 48]*;
let arr_lit_large_p: [i32; 0x8000]* = arr_lit_large;

// --- fail_literal_runtime.carbon

library "[[@TEST_NAME]]";

fn AndLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";

fn F(a: Core.IntLiteral()) -> Core.IntLiteral() {
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE+6]]:10: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
  // CHECK:STDERR:   return AndLit(5, a);
  // CHECK:STDERR:          ^~~~~~~~~~~~
  // CHECK:STDERR: fail_literal_runtime.carbon:[[@LINE-6]]:1: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
  // CHECK:STDERR: fn AndLit(a: Core.IntLiteral(), b: Core.IntLiteral()) -> Core.IntLiteral() = "int.and";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return AndLit(5, a);
}

// CHECK:STDOUT: --- int_and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %int_12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_10.1: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_12.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_10.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_10.2: %i32 = int_value 10 [template]
// CHECK:STDOUT:   %int_8.1: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %Convert.type.11: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.11: %Convert.type.11 = struct_value () [template]
// CHECK:STDOUT:   %interface.6: <witness> = interface_witness (%Convert.11) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_8.1, %Convert.11 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %int_8.2: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_8.2, %i32 [template]
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
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, %And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.1]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10.1]
// CHECK:STDOUT:   %impl.elem0.loc6_20: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_20: <bound method> = bound_method %int_12, %impl.elem0.loc6_20 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_20: <specific function> = specific_function %Convert.bound.loc6_20, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_20: init %i32 = call %Convert.specific_fn.loc6_20(%int_12) [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc6_20.1: %i32 = value_of_initializer %int.convert_checked.loc6_20 [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc6_20.2: %i32 = converted %int_12, %.loc6_20.1 [template = constants.%int_12.2]
// CHECK:STDOUT:   %impl.elem0.loc6_24: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc6_24: <bound method> = bound_method %int_10, %impl.elem0.loc6_24 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_24: <specific function> = specific_function %Convert.bound.loc6_24, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc6_24: init %i32 = call %Convert.specific_fn.loc6_24(%int_10) [template = constants.%int_10.2]
// CHECK:STDOUT:   %.loc6_24.1: %i32 = value_of_initializer %int.convert_checked.loc6_24 [template = constants.%int_10.2]
// CHECK:STDOUT:   %.loc6_24.2: %i32 = converted %int_10, %.loc6_24.1 [template = constants.%int_10.2]
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%.loc6_20.2, %.loc6_24.2) [template = constants.%int_8.1]
// CHECK:STDOUT:   %impl.elem0.loc6_26: %Convert.type.5 = interface_witness_access constants.%interface.6, element0 [template = constants.%Convert.11]
// CHECK:STDOUT:   %Convert.bound.loc6_26: <bound method> = bound_method %int.and, %impl.elem0.loc6_26 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc6_26: <specific function> = specific_function %Convert.bound.loc6_26, @Convert.3(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %.loc6_26.1: %i32 = value_of_initializer %int.and [template = constants.%int_8.1]
// CHECK:STDOUT:   %.loc6_26.2: %i32 = converted %int.and, %.loc6_26.1 [template = constants.%int_8.1]
// CHECK:STDOUT:   %int.convert_checked.loc6_26: init Core.IntLiteral = call %Convert.specific_fn.loc6_26(%.loc6_26.2) [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc6_26.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc6_26 [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc6_26.4: Core.IntLiteral = converted %int.and, %.loc6_26.3 [template = constants.%int_8.2]
// CHECK:STDOUT:   %array_type.loc6: type = array_type %.loc6_26.4, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %arr.var: ref %array_type = var arr
// CHECK:STDOUT:   %arr: ref %array_type = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8.2]
// CHECK:STDOUT:   %array_type.loc7: type = array_type %int_8, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc9_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc9_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc9_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, file.%And.decl [template = constants.%And]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %int.and
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int.and, %.loc10_19.1
// CHECK:STDOUT:   return %.loc10_19.2
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
// CHECK:STDOUT:   %AndLit.type: type = fn_type @AndLit [template]
// CHECK:STDOUT:   %AndLit: %AndLit.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_8, %i32 [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %array_type.1 [template]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Op.type.1: type = fn_type @Op.1 [template]
// CHECK:STDOUT:   %Op.type.2: type = fn_type @Op.2 [template]
// CHECK:STDOUT:   %Op.2: %Op.type.2 = struct_value () [template]
// CHECK:STDOUT:   %interface.1: <witness> = interface_witness (%Op.2) [template]
// CHECK:STDOUT:   %Op.bound.1: <bound method> = bound_method %int_1, %Op.2 [template]
// CHECK:STDOUT:   %int_-1: Core.IntLiteral = int_value -1 [template]
// CHECK:STDOUT:   %int_48: Core.IntLiteral = int_value 48 [template]
// CHECK:STDOUT:   %Op.type.3: type = fn_type @Op.3 [template]
// CHECK:STDOUT:   %Op.type.4: type = fn_type @Op.4 [template]
// CHECK:STDOUT:   %Op.4: %Op.type.4 = struct_value () [template]
// CHECK:STDOUT:   %interface.2: <witness> = interface_witness (%Op.4) [template]
// CHECK:STDOUT:   %Op.bound.2: <bound method> = bound_method %int_9223372036854775808, %Op.4 [template]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_32768, %i32 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %array_type.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .IntLiteral = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .Negate = %import_ref.6
// CHECK:STDOUT:     .RightShift = %import_ref.44
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [template = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .AndLit = %AndLit.decl
// CHECK:STDOUT:     .arr_lit = %arr_lit
// CHECK:STDOUT:     .arr_lit_p = @__global_init.%arr_lit_p
// CHECK:STDOUT:     .arr_lit_large = %arr_lit_large
// CHECK:STDOUT:     .arr_lit_large_p = @__global_init.%arr_lit_large_p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AndLit.decl: %AndLit.type = fn_decl @AndLit [template = constants.%AndLit] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_14: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_18: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_30: init type = call %IntLiteral.ref.loc4_18() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_30.1: type = value_of_initializer %int_literal.make_type.loc4_30 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_30.2: type = converted %int_literal.make_type.loc4_30, %.loc4_30.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_36: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_40: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_52: init type = call %IntLiteral.ref.loc4_40() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_52.1: type = value_of_initializer %int_literal.make_type.loc4_52 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_52.2: type = converted %int_literal.make_type.loc4_52, %.loc4_52.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_58: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_62: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref.1 [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_74: init type = call %IntLiteral.ref.loc4_62() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_74.1: type = value_of_initializer %int_literal.make_type.loc4_74 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_74.2: type = converted %int_literal.make_type.loc4_74, %.loc4_74.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %AndLit.ref.loc6: %AndLit.type = name_ref AndLit, %AndLit.decl [template = constants.%AndLit]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %int.and.loc6: init Core.IntLiteral = call %AndLit.ref.loc6(%int_12, %int_10) [template = constants.%int_8]
// CHECK:STDOUT:   %.loc6_33.1: Core.IntLiteral = value_of_initializer %int.and.loc6 [template = constants.%int_8]
// CHECK:STDOUT:   %.loc6_33.2: Core.IntLiteral = converted %int.and.loc6, %.loc6_33.1 [template = constants.%int_8]
// CHECK:STDOUT:   %array_type.loc6: type = array_type %.loc6_33.2, %i32 [template = constants.%array_type.1]
// CHECK:STDOUT:   %arr_lit.var: ref %array_type.1 = var arr_lit
// CHECK:STDOUT:   %arr_lit: ref %array_type.1 = bind_name arr_lit, %arr_lit.var
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8]
// CHECK:STDOUT:   %array_type.loc7: type = array_type %int_8, %i32 [template = constants.%array_type.1]
// CHECK:STDOUT:   %ptr.loc7: type = ptr_type %array_type.1 [template = constants.%ptr.1]
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %AndLit.ref.loc10: %AndLit.type = name_ref AndLit, %AndLit.decl [template = constants.%AndLit]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc10_56: %Op.type.1 = interface_witness_access constants.%interface.1, element0 [template = constants.%Op.2]
// CHECK:STDOUT:   %Op.bound.loc10_56: <bound method> = bound_method %int_1, %impl.elem0.loc10_56 [template = constants.%Op.bound.1]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %Op.bound.loc10_56(%int_1) [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_56.1: Core.IntLiteral = value_of_initializer %int.snegate [template = constants.%int_-1]
// CHECK:STDOUT:   %.loc10_56.2: Core.IntLiteral = converted %int.snegate, %.loc10_56.1 [template = constants.%int_-1]
// CHECK:STDOUT:   %int.and.loc10: init Core.IntLiteral = call %AndLit.ref.loc10(%int_9223372036854775808, %.loc10_56.2) [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %int_48: Core.IntLiteral = int_value 48 [template = constants.%int_48]
// CHECK:STDOUT:   %impl.elem0.loc10_60: %Op.type.3 = interface_witness_access constants.%interface.2, element0 [template = constants.%Op.4]
// CHECK:STDOUT:   %Op.bound.loc10_60: <bound method> = bound_method %int.and.loc10, %impl.elem0.loc10_60 [template = constants.%Op.bound.2]
// CHECK:STDOUT:   %.loc10_58.1: Core.IntLiteral = value_of_initializer %int.and.loc10 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %.loc10_58.2: Core.IntLiteral = converted %int.and.loc10, %.loc10_58.1 [template = constants.%int_9223372036854775808]
// CHECK:STDOUT:   %int.right_shift: init Core.IntLiteral = call %Op.bound.loc10_60(%.loc10_58.2, %int_48) [template = constants.%int_32768]
// CHECK:STDOUT:   %.loc10_60.1: Core.IntLiteral = value_of_initializer %int.right_shift [template = constants.%int_32768]
// CHECK:STDOUT:   %.loc10_60.2: Core.IntLiteral = converted %int.right_shift, %.loc10_60.1 [template = constants.%int_32768]
// CHECK:STDOUT:   %array_type.loc10: type = array_type %.loc10_60.2, %i32 [template = constants.%array_type.2]
// CHECK:STDOUT:   %ptr.loc10: type = ptr_type %array_type.2 [template = constants.%ptr.2]
// CHECK:STDOUT:   %arr_lit_large.var: ref %ptr.2 = var arr_lit_large
// CHECK:STDOUT:   %arr_lit_large: ref %ptr.2 = bind_name arr_lit_large, %arr_lit_large.var
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [template = constants.%int_32768]
// CHECK:STDOUT:   %array_type.loc11: type = array_type %int_32768, %i32 [template = constants.%array_type.2]
// CHECK:STDOUT:   %ptr.loc11: type = ptr_type %array_type.2 [template = constants.%ptr.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AndLit(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr_lit.ref: ref %array_type.1 = name_ref arr_lit, file.%arr_lit
// CHECK:STDOUT:   %addr: %ptr.1 = addr_of %arr_lit.ref
// CHECK:STDOUT:   %arr_lit_p: %ptr.1 = bind_name arr_lit_p, %addr
// CHECK:STDOUT:   %arr_lit_large.ref: ref %ptr.2 = name_ref arr_lit_large, file.%arr_lit_large
// CHECK:STDOUT:   %.loc11: %ptr.2 = bind_value %arr_lit_large.ref
// CHECK:STDOUT:   %arr_lit_large_p: %ptr.2 = bind_name arr_lit_large_p, %.loc11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_literal_runtime.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [template]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [template]
// CHECK:STDOUT:   %AndLit.type: type = fn_type @AndLit [template]
// CHECK:STDOUT:   %AndLit: %AndLit.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
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
// CHECK:STDOUT:     .AndLit = %AndLit.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AndLit.decl: %AndLit.type = fn_decl @AndLit [template = constants.%AndLit] {
// CHECK:STDOUT:     %a.patt: Core.IntLiteral = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: Core.IntLiteral = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: Core.IntLiteral = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: Core.IntLiteral = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: Core.IntLiteral = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: Core.IntLiteral = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Core.ref.loc4_14: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_18: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_30: init type = call %IntLiteral.ref.loc4_18() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_30.1: type = value_of_initializer %int_literal.make_type.loc4_30 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_30.2: type = converted %int_literal.make_type.loc4_30, %.loc4_30.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_36: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_40: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_52: init type = call %IntLiteral.ref.loc4_40() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_52.1: type = value_of_initializer %int_literal.make_type.loc4_52 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_52.2: type = converted %int_literal.make_type.loc4_52, %.loc4_52.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %Core.ref.loc4_58: <namespace> = name_ref Core, imports.%Core [template = imports.%Core]
// CHECK:STDOUT:     %IntLiteral.ref.loc4_62: %IntLiteral.type = name_ref IntLiteral, imports.%import_ref [template = constants.%IntLiteral]
// CHECK:STDOUT:     %int_literal.make_type.loc4_74: init type = call %IntLiteral.ref.loc4_62() [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_74.1: type = value_of_initializer %int_literal.make_type.loc4_74 [template = Core.IntLiteral]
// CHECK:STDOUT:     %.loc4_74.2: type = converted %int_literal.make_type.loc4_74, %.loc4_74.1 [template = Core.IntLiteral]
// CHECK:STDOUT:     %a.param: Core.IntLiteral = value_param runtime_param0
// CHECK:STDOUT:     %a: Core.IntLiteral = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: Core.IntLiteral = value_param runtime_param1
// CHECK:STDOUT:     %b: Core.IntLiteral = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param runtime_param2
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
// CHECK:STDOUT: fn @AndLit(%a.param_patt: Core.IntLiteral, %b.param_patt: Core.IntLiteral) -> Core.IntLiteral = "int.and";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param_patt: Core.IntLiteral) -> Core.IntLiteral {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %AndLit.ref: %AndLit.type = name_ref AndLit, file.%AndLit.decl [template = constants.%AndLit]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %a.ref: Core.IntLiteral = name_ref a, %a
// CHECK:STDOUT:   %int.and: init Core.IntLiteral = call %AndLit.ref(%int_5, %a.ref)
// CHECK:STDOUT:   %.loc13_22.1: Core.IntLiteral = value_of_initializer %int.and
// CHECK:STDOUT:   %.loc13_22.2: Core.IntLiteral = converted %int.and, %.loc13_22.1
// CHECK:STDOUT:   return %.loc13_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
