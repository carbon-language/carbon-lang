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

fn And(a: i32, b: i32) -> i32 = "int.and";

// CHECK:STDERR: int_and.carbon:[[@LINE+3]]:16: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr: [i32; And(12, 10)];
// CHECK:STDERR:                ^~~~~~~~~~~
var arr: [i32; And(12, 10)];
let arr_p: [i32; 8]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return And(a, b);
}

// CHECK:STDOUT: --- int_and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_12, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_12) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_10, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_10) [symbolic]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_8, %i32 [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:     %int_32.loc2_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_11: init type = call constants.%Int(%int_32.loc2_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_signed.loc2_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_signed.loc2_11, %.loc2_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_19: init type = call constants.%Int(%int_32.loc2_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_signed.loc2_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_signed.loc2_19, %.loc2_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_27: init type = call constants.%Int(%int_32.loc2_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_signed.loc2_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_signed.loc2_27, %.loc2_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, %And.decl [template = constants.%And]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %impl.elem0.loc7_20: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_20: <bound method> = bound_method %int_12, %impl.elem0.loc7_20 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_20: <specific function> = specific_function %Convert.bound.loc7_20, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc7_20: init %iN = call %Convert.specific_fn.loc7_20(%int_12) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_20.1: %i32 = value_of_initializer %int.convert_checked.loc7_20 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_20.2: %i32 = converted %int_12, %.loc7_20.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc7_24: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_24: <bound method> = bound_method %int_10, %impl.elem0.loc7_24 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_24: <specific function> = specific_function %Convert.bound.loc7_24, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc7_24: init %iN = call %Convert.specific_fn.loc7_24(%int_10) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_24.1: %i32 = value_of_initializer %int.convert_checked.loc7_24 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_24.2: %i32 = converted %int_10, %.loc7_24.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%.loc7_20.2, %.loc7_24.2)
// CHECK:STDOUT:   %.loc7_11.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_11.2: type = converted %int.make_type_signed.loc7, %.loc7_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [template = constants.%int_8]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_signed.loc8, %.loc8_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_8, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc10_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_19: init type = call constants.%Int(%int_32.loc10_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_19.1: type = value_of_initializer %int.make_type_signed.loc10_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_19.2: type = converted %int.make_type_signed.loc10_19, %.loc10_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_27: init type = call constants.%Int(%int_32.loc10_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_27.1: type = value_of_initializer %int.make_type_signed.loc10_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_27.2: type = converted %int.make_type_signed.loc10_27, %.loc10_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc10_35: init type = call constants.%Int(%int_32.loc10_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_35.1: type = value_of_initializer %int.make_type_signed.loc10_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc10_35.2: type = converted %int.make_type_signed.loc10_35, %.loc10_35.1 [template = constants.%i32]
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
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.and
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int.and, %.loc11_19.1
// CHECK:STDOUT:   return %.loc11_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref <error> = name_ref arr, file.%arr
// CHECK:STDOUT:   %addr: <error> = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %ptr = bind_name arr_p, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
