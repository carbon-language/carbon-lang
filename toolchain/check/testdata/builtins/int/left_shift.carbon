// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/left_shift.carbon

// --- int_left_shift.carbon

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";

// CHECK:STDERR: int_left_shift.carbon:[[@LINE+3]]:16: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr: [i32; LeftShift(5, 2)];
// CHECK:STDERR:                ^~~~~~~~~~~~~~~
var arr: [i32; LeftShift(5, 2)];
let arr_p: [i32; 20]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return LeftShift(a, b);
}

// TODO: Test mixed types for LHS and RHS.

// --- fail_bad_shift.carbon

package BadShift;

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn Negate(a: i32) -> i32 = "int.snegate";

// Shift greater than size is disallowed.
let size_1: i32 = LeftShift(1, 31);
let size_2: i32 = LeftShift(1, 32);
let size_3: i32 = LeftShift(1, 33);

// Overflow is allowed if the shift distance is in bounds.
let overflow_1: i32 = LeftShift(1000, 31);
let overflow_2: i32 = LeftShift(1000, 32);

// Oversize shifts aren't allowed even if there's no overflow.
let no_overflow_1: i32 = LeftShift(0, 31);
let no_overflow_2: i32 = LeftShift(0, 32);

// Negative shifts aren't allowed either.
let negative: i32 = LeftShift(1, Negate(1));

// CHECK:STDOUT: --- int_left_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_5, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_5) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_2) [symbolic]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_20, %i32 [template]
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
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.1 = name_ref LeftShift, %LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %impl.elem0.loc7_26: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_26: <bound method> = bound_method %int_5, %impl.elem0.loc7_26 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_26: <specific function> = specific_function %Convert.bound.loc7_26, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc7_26: init %Int = call %Convert.specific_fn.loc7_26(%int_5) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_26.1: %i32 = value_of_initializer %int.convert_checked.loc7_26 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_26.2: %i32 = converted %int_5, %.loc7_26.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc7_29: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_29: <bound method> = bound_method %int_2, %impl.elem0.loc7_29 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_29: <specific function> = specific_function %Convert.bound.loc7_29, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc7_29: init %Int = call %Convert.specific_fn.loc7_29(%int_2) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_29.1: %i32 = value_of_initializer %int.convert_checked.loc7_29 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_29.2: %i32 = converted %int_2, %.loc7_29.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%.loc7_26.2, %.loc7_29.2)
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [template = constants.%int_20]
// CHECK:STDOUT:   %array_type: type = array_type %int_20, %i32 [template = constants.%array_type]
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
// CHECK:STDOUT:     %i32.loc10_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc10_35: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10_35: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.left_shift: init %i32 = call %LeftShift.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc11_25.1: %i32 = value_of_initializer %int.left_shift
// CHECK:STDOUT:   %.loc11_25.2: %i32 = converted %int.left_shift, %.loc11_25.1
// CHECK:STDOUT:   return %.loc11_25.2
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
// CHECK:STDOUT: --- fail_bad_shift.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %LeftShift.type.1: type = fn_type @LeftShift.1 [template]
// CHECK:STDOUT:   %LeftShift: %LeftShift.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_31, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_31) [symbolic]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_32, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.3: init %Int = call %Convert.specific_fn.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_33, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.4: init %Int = call %Convert.specific_fn.4(%int_33) [symbolic]
// CHECK:STDOUT:   %int_1000: Core.IntLiteral = int_value 1000 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_1000, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.5: init %Int = call %Convert.specific_fn.5(%int_1000) [symbolic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.6: init %Int = call %Convert.specific_fn.6(%int_0) [symbolic]
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
// CHECK:STDOUT:     .LeftShift = %LeftShift.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .size_1 = @__global_init.%size_1
// CHECK:STDOUT:     .size_2 = @__global_init.%size_2
// CHECK:STDOUT:     .size_3 = @__global_init.%size_3
// CHECK:STDOUT:     .overflow_1 = @__global_init.%overflow_1
// CHECK:STDOUT:     .overflow_2 = @__global_init.%overflow_2
// CHECK:STDOUT:     .no_overflow_1 = @__global_init.%no_overflow_1
// CHECK:STDOUT:     .no_overflow_2 = @__global_init.%no_overflow_2
// CHECK:STDOUT:     .negative = @__global_init.%negative
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %LeftShift.decl: %LeftShift.type.1 = fn_decl @LeftShift.1 [template = constants.%LeftShift] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc4_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_25: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc4_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc4_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @LeftShift.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.left_shift";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %LeftShift.ref.loc8: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_31.loc8: Core.IntLiteral = int_value 31 [template = constants.%int_31]
// CHECK:STDOUT:   %impl.elem0.loc8_29: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_29: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_29 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_29: <specific function> = specific_function %Convert.bound.loc8_29, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_29: init %Int = call %Convert.specific_fn.loc8_29(%int_1.loc8) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_29.1: %i32 = value_of_initializer %int.convert_checked.loc8_29 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_29.2: %i32 = converted %int_1.loc8, %.loc8_29.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc8_32: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_32: <bound method> = bound_method %int_31.loc8, %impl.elem0.loc8_32 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_32: <specific function> = specific_function %Convert.bound.loc8_32, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_32: init %Int = call %Convert.specific_fn.loc8_32(%int_31.loc8) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_32.1: %i32 = value_of_initializer %int.convert_checked.loc8_32 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_32.2: %i32 = converted %int_31.loc8, %.loc8_32.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.left_shift.loc8: init %i32 = call %LeftShift.ref.loc8(%.loc8_29.2, %.loc8_32.2)
// CHECK:STDOUT:   %.loc8_35.1: %i32 = value_of_initializer %int.left_shift.loc8
// CHECK:STDOUT:   %.loc8_35.2: %i32 = converted %int.left_shift.loc8, %.loc8_35.1
// CHECK:STDOUT:   %size_1: %i32 = bind_name size_1, %.loc8_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc9: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %impl.elem0.loc9_29: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_29: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_29 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_29: <specific function> = specific_function %Convert.bound.loc9_29, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_29: init %Int = call %Convert.specific_fn.loc9_29(%int_1.loc9) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_29.1: %i32 = value_of_initializer %int.convert_checked.loc9_29 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_29.2: %i32 = converted %int_1.loc9, %.loc9_29.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc9_32: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_32: <bound method> = bound_method %int_32.loc9, %impl.elem0.loc9_32 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_32: <specific function> = specific_function %Convert.bound.loc9_32, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc9_32: init %Int = call %Convert.specific_fn.loc9_32(%int_32.loc9) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_32.1: %i32 = value_of_initializer %int.convert_checked.loc9_32 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc9_32.2: %i32 = converted %int_32.loc9, %.loc9_32.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.left_shift.loc9: init %i32 = call %LeftShift.ref.loc9(%.loc9_29.2, %.loc9_32.2)
// CHECK:STDOUT:   %.loc9_35.1: %i32 = value_of_initializer %int.left_shift.loc9
// CHECK:STDOUT:   %.loc9_35.2: %i32 = converted %int.left_shift.loc9, %.loc9_35.1
// CHECK:STDOUT:   %size_2: %i32 = bind_name size_2, %.loc9_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc10: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [template = constants.%int_33]
// CHECK:STDOUT:   %impl.elem0.loc10_29: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc10_29: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_29 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_29: <specific function> = specific_function %Convert.bound.loc10_29, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc10_29: init %Int = call %Convert.specific_fn.loc10_29(%int_1.loc10) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc10_29.1: %i32 = value_of_initializer %int.convert_checked.loc10_29 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc10_29.2: %i32 = converted %int_1.loc10, %.loc10_29.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc10_32: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc10_32: <bound method> = bound_method %int_33, %impl.elem0.loc10_32 [symbolic = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_32: <specific function> = specific_function %Convert.bound.loc10_32, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc10_32: init %Int = call %Convert.specific_fn.loc10_32(%int_33) [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc10_32.1: %i32 = value_of_initializer %int.convert_checked.loc10_32 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %.loc10_32.2: %i32 = converted %int_33, %.loc10_32.1 [symbolic = constants.%int.convert_checked.4]
// CHECK:STDOUT:   %int.left_shift.loc10: init %i32 = call %LeftShift.ref.loc10(%.loc10_29.2, %.loc10_32.2)
// CHECK:STDOUT:   %.loc10_35.1: %i32 = value_of_initializer %int.left_shift.loc10
// CHECK:STDOUT:   %.loc10_35.2: %i32 = converted %int.left_shift.loc10, %.loc10_35.1
// CHECK:STDOUT:   %size_3: %i32 = bind_name size_3, %.loc10_35.2
// CHECK:STDOUT:   %LeftShift.ref.loc13: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1000.loc13: Core.IntLiteral = int_value 1000 [template = constants.%int_1000]
// CHECK:STDOUT:   %int_31.loc13: Core.IntLiteral = int_value 31 [template = constants.%int_31]
// CHECK:STDOUT:   %impl.elem0.loc13_33: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_33: <bound method> = bound_method %int_1000.loc13, %impl.elem0.loc13_33 [symbolic = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_33: <specific function> = specific_function %Convert.bound.loc13_33, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc13_33: init %Int = call %Convert.specific_fn.loc13_33(%int_1000.loc13) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc13_33.1: %i32 = value_of_initializer %int.convert_checked.loc13_33 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc13_33.2: %i32 = converted %int_1000.loc13, %.loc13_33.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %impl.elem0.loc13_39: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc13_39: <bound method> = bound_method %int_31.loc13, %impl.elem0.loc13_39 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_39: <specific function> = specific_function %Convert.bound.loc13_39, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_39: init %Int = call %Convert.specific_fn.loc13_39(%int_31.loc13) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc13_39.1: %i32 = value_of_initializer %int.convert_checked.loc13_39 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc13_39.2: %i32 = converted %int_31.loc13, %.loc13_39.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.left_shift.loc13: init %i32 = call %LeftShift.ref.loc13(%.loc13_33.2, %.loc13_39.2)
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %int.left_shift.loc13
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int.left_shift.loc13, %.loc13_42.1
// CHECK:STDOUT:   %overflow_1: %i32 = bind_name overflow_1, %.loc13_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc14: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1000.loc14: Core.IntLiteral = int_value 1000 [template = constants.%int_1000]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %impl.elem0.loc14_33: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_33: <bound method> = bound_method %int_1000.loc14, %impl.elem0.loc14_33 [symbolic = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_33: <specific function> = specific_function %Convert.bound.loc14_33, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc14_33: init %Int = call %Convert.specific_fn.loc14_33(%int_1000.loc14) [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc14_33.1: %i32 = value_of_initializer %int.convert_checked.loc14_33 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %.loc14_33.2: %i32 = converted %int_1000.loc14, %.loc14_33.1 [symbolic = constants.%int.convert_checked.5]
// CHECK:STDOUT:   %impl.elem0.loc14_39: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc14_39: <bound method> = bound_method %int_32.loc14, %impl.elem0.loc14_39 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_39: <specific function> = specific_function %Convert.bound.loc14_39, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc14_39: init %Int = call %Convert.specific_fn.loc14_39(%int_32.loc14) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc14_39.1: %i32 = value_of_initializer %int.convert_checked.loc14_39 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc14_39.2: %i32 = converted %int_32.loc14, %.loc14_39.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.left_shift.loc14: init %i32 = call %LeftShift.ref.loc14(%.loc14_33.2, %.loc14_39.2)
// CHECK:STDOUT:   %.loc14_42.1: %i32 = value_of_initializer %int.left_shift.loc14
// CHECK:STDOUT:   %.loc14_42.2: %i32 = converted %int.left_shift.loc14, %.loc14_42.1
// CHECK:STDOUT:   %overflow_2: %i32 = bind_name overflow_2, %.loc14_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc17: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc17: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_31.loc17: Core.IntLiteral = int_value 31 [template = constants.%int_31]
// CHECK:STDOUT:   %impl.elem0.loc17_36: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc17_36: <bound method> = bound_method %int_0.loc17, %impl.elem0.loc17_36 [symbolic = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_36: <specific function> = specific_function %Convert.bound.loc17_36, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc17_36: init %Int = call %Convert.specific_fn.loc17_36(%int_0.loc17) [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc17_36.1: %i32 = value_of_initializer %int.convert_checked.loc17_36 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc17_36.2: %i32 = converted %int_0.loc17, %.loc17_36.1 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %impl.elem0.loc17_39: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc17_39: <bound method> = bound_method %int_31.loc17, %impl.elem0.loc17_39 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17_39: <specific function> = specific_function %Convert.bound.loc17_39, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17_39: init %Int = call %Convert.specific_fn.loc17_39(%int_31.loc17) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc17_39.1: %i32 = value_of_initializer %int.convert_checked.loc17_39 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc17_39.2: %i32 = converted %int_31.loc17, %.loc17_39.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.left_shift.loc17: init %i32 = call %LeftShift.ref.loc17(%.loc17_36.2, %.loc17_39.2)
// CHECK:STDOUT:   %.loc17_42.1: %i32 = value_of_initializer %int.left_shift.loc17
// CHECK:STDOUT:   %.loc17_42.2: %i32 = converted %int.left_shift.loc17, %.loc17_42.1
// CHECK:STDOUT:   %no_overflow_1: %i32 = bind_name no_overflow_1, %.loc17_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc18: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %impl.elem0.loc18_36: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_36: <bound method> = bound_method %int_0.loc18, %impl.elem0.loc18_36 [symbolic = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_36: <specific function> = specific_function %Convert.bound.loc18_36, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc18_36: init %Int = call %Convert.specific_fn.loc18_36(%int_0.loc18) [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc18_36.1: %i32 = value_of_initializer %int.convert_checked.loc18_36 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %.loc18_36.2: %i32 = converted %int_0.loc18, %.loc18_36.1 [symbolic = constants.%int.convert_checked.6]
// CHECK:STDOUT:   %impl.elem0.loc18_39: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc18_39: <bound method> = bound_method %int_32.loc18, %impl.elem0.loc18_39 [symbolic = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_39: <specific function> = specific_function %Convert.bound.loc18_39, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc18_39: init %Int = call %Convert.specific_fn.loc18_39(%int_32.loc18) [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc18_39.1: %i32 = value_of_initializer %int.convert_checked.loc18_39 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %.loc18_39.2: %i32 = converted %int_32.loc18, %.loc18_39.1 [symbolic = constants.%int.convert_checked.3]
// CHECK:STDOUT:   %int.left_shift.loc18: init %i32 = call %LeftShift.ref.loc18(%.loc18_36.2, %.loc18_39.2)
// CHECK:STDOUT:   %.loc18_42.1: %i32 = value_of_initializer %int.left_shift.loc18
// CHECK:STDOUT:   %.loc18_42.2: %i32 = converted %int.left_shift.loc18, %.loc18_42.1
// CHECK:STDOUT:   %no_overflow_2: %i32 = bind_name no_overflow_2, %.loc18_42.2
// CHECK:STDOUT:   %LeftShift.ref.loc21: %LeftShift.type.1 = name_ref LeftShift, file.%LeftShift.decl [template = constants.%LeftShift]
// CHECK:STDOUT:   %int_1.loc21_31: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %Negate.ref: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc21_41: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc21_41: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc21_41: <bound method> = bound_method %int_1.loc21_41, %impl.elem0.loc21_41 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_41: <specific function> = specific_function %Convert.bound.loc21_41, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_41: init %Int = call %Convert.specific_fn.loc21_41(%int_1.loc21_41) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc21_41.1: %i32 = value_of_initializer %int.convert_checked.loc21_41 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc21_41.2: %i32 = converted %int_1.loc21_41, %.loc21_41.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate: init %i32 = call %Negate.ref(%.loc21_41.2)
// CHECK:STDOUT:   %impl.elem0.loc21_31: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc21_31: <bound method> = bound_method %int_1.loc21_31, %impl.elem0.loc21_31 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_31: <specific function> = specific_function %Convert.bound.loc21_31, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_31: init %Int = call %Convert.specific_fn.loc21_31(%int_1.loc21_31) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc21_31.1: %i32 = value_of_initializer %int.convert_checked.loc21_31 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc21_31.2: %i32 = converted %int_1.loc21_31, %.loc21_31.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc21_42.1: %i32 = value_of_initializer %int.snegate
// CHECK:STDOUT:   %.loc21_42.2: %i32 = converted %int.snegate, %.loc21_42.1
// CHECK:STDOUT:   %int.left_shift.loc21: init %i32 = call %LeftShift.ref.loc21(%.loc21_31.2, %.loc21_42.2)
// CHECK:STDOUT:   %.loc21_44.1: %i32 = value_of_initializer %int.left_shift.loc21
// CHECK:STDOUT:   %.loc21_44.2: %i32 = converted %int.left_shift.loc21, %.loc21_44.1
// CHECK:STDOUT:   %negative: %i32 = bind_name negative, %.loc21_44.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
