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

fn Complement(a: i32) -> i32 = "int.complement";
fn And(a: i32, b: i32) -> i32 = "int.and";

// CHECK:STDERR: int_complement.carbon:[[@LINE+3]]:16: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr: [i32; And(Complement(0x123456), 0xFFFFFF)];
// CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var arr: [i32; And(Complement(0x123456), 0xFFFFFF)];
let arr_p: [i32; 0xEDCBA9]* = &arr;

fn RuntimeCall(a: i32) -> i32 {
  return Complement(a);
}

// CHECK:STDOUT: --- int_complement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Complement.type: type = fn_type @Complement [template]
// CHECK:STDOUT:   %Complement: %Complement.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %int_1193046: Core.IntLiteral = int_value 1193046 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %iN: type = int_type signed, %N [symbolic]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%Convert.3) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1193046, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %iN = call %Convert.specific_fn.1(%int_1193046) [symbolic]
// CHECK:STDOUT:   %int_16777215: Core.IntLiteral = int_value 16777215 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_16777215, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %iN = call %Convert.specific_fn.2(%int_16777215) [symbolic]
// CHECK:STDOUT:   %int_15584169: Core.IntLiteral = int_value 15584169 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_15584169, %i32 [template]
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
// CHECK:STDOUT:     %int_32.loc2_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_18: init type = call constants.%Int(%int_32.loc2_18) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.1: type = value_of_initializer %int.make_type_signed.loc2_18 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_18.2: type = converted %int.make_type_signed.loc2_18, %.loc2_18.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_26: init type = call constants.%Int(%int_32.loc2_26) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.1: type = value_of_initializer %int.make_type_signed.loc2_26 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_26.2: type = converted %int.make_type_signed.loc2_26, %.loc2_26.1 [template = constants.%i32]
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
// CHECK:STDOUT:     %int_32.loc3_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_11: init type = call constants.%Int(%int_32.loc3_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_11.1: type = value_of_initializer %int.make_type_signed.loc3_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_11.2: type = converted %int.make_type_signed.loc3_11, %.loc3_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc3_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_19: init type = call constants.%Int(%int_32.loc3_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_19.1: type = value_of_initializer %int.make_type_signed.loc3_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_19.2: type = converted %int.make_type_signed.loc3_19, %.loc3_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc3_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc3_27: init type = call constants.%Int(%int_32.loc3_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_27.1: type = value_of_initializer %int.make_type_signed.loc3_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc3_27.2: type = converted %int.make_type_signed.loc3_27, %.loc3_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %And.ref: %And.type = name_ref And, %And.decl [template = constants.%And]
// CHECK:STDOUT:   %Complement.ref: %Complement.type = name_ref Complement, %Complement.decl [template = constants.%Complement]
// CHECK:STDOUT:   %int_1193046: Core.IntLiteral = int_value 1193046 [template = constants.%int_1193046]
// CHECK:STDOUT:   %impl.elem0.loc8_31: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_31: <bound method> = bound_method %int_1193046, %impl.elem0.loc8_31 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_31: <specific function> = specific_function %Convert.bound.loc8_31, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_31: init %iN = call %Convert.specific_fn.loc8_31(%int_1193046) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_31.1: %i32 = value_of_initializer %int.convert_checked.loc8_31 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_31.2: %i32 = converted %int_1193046, %.loc8_31.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.complement: init %i32 = call %Complement.ref(%.loc8_31.2)
// CHECK:STDOUT:   %int_16777215: Core.IntLiteral = int_value 16777215 [template = constants.%int_16777215]
// CHECK:STDOUT:   %.loc8_39.1: %i32 = value_of_initializer %int.complement
// CHECK:STDOUT:   %.loc8_39.2: %i32 = converted %int.complement, %.loc8_39.1
// CHECK:STDOUT:   %impl.elem0.loc8_42: %Convert.type.2 = impl_witness_access constants.%impl_witness, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_42: <bound method> = bound_method %int_16777215, %impl.elem0.loc8_42 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_42: <specific function> = specific_function %Convert.bound.loc8_42, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_42: init %iN = call %Convert.specific_fn.loc8_42(%int_16777215) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_42.1: %i32 = value_of_initializer %int.convert_checked.loc8_42 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_42.2: %i32 = converted %int_16777215, %.loc8_42.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.and: init %i32 = call %And.ref(%.loc8_39.2, %.loc8_42.2)
// CHECK:STDOUT:   %.loc8_11.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_11.2: type = converted %int.make_type_signed.loc8, %.loc8_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%int_32.loc9) [template = constants.%i32]
// CHECK:STDOUT:   %int_15584169: Core.IntLiteral = int_value 15584169 [template = constants.%int_15584169]
// CHECK:STDOUT:   %.loc9_13.1: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_13.2: type = converted %int.make_type_signed.loc9, %.loc9_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_15584169, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %ptr: type = ptr_type %array_type [template = constants.%ptr]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_19: init type = call constants.%Int(%int_32.loc11_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.1: type = value_of_initializer %int.make_type_signed.loc11_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_19.2: type = converted %int.make_type_signed.loc11_19, %.loc11_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_27: init type = call constants.%Int(%int_32.loc11_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_27.1: type = value_of_initializer %int.make_type_signed.loc11_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_27.2: type = converted %int.make_type_signed.loc11_27, %.loc11_27.1 [template = constants.%i32]
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
// CHECK:STDOUT:   %.loc12_23.1: %i32 = value_of_initializer %int.complement
// CHECK:STDOUT:   %.loc12_23.2: %i32 = converted %int.complement, %.loc12_23.1
// CHECK:STDOUT:   return %.loc12_23.2
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
