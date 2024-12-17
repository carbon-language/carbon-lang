// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/smod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/smod.carbon

// --- int_div.carbon

fn Mod(a: i32, b: i32) -> i32 = "int.smod";

// CHECK:STDERR: int_div.carbon:[[@LINE+3]]:16: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var arr: [i32; Mod(5, 3)];
// CHECK:STDERR:                ^~~~~~~~~
var arr: [i32; Mod(5, 3)];
let arr_p: [i32; 2]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Mod(a, b);
}

// --- fail_overflow.carbon

package FailOverflow;

fn Mod(a: i32, b: i32) -> i32 = "int.smod";
fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn Negate(a: i32) -> i32 = "int.snegate";

// -0x7FFF_FFFF % -1 is OK.
let a: i32 = Mod(Negate(0x7FFF_FFFF), Negate(1));

// -0x8000_0000 % 1 is OK.
let b: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), 1);

// -0x8000_0000 / -1 overflows, so -0x8000_0000 % -1 is disallowed, even though
// its result is representable.
let c: i32 = Mod(Sub(Negate(0x7FFF_FFFF), 1), Negate(1));

// --- fail_div_by_zero.carbon

package FailDivByZero;

fn Mod(a: i32, b: i32) -> i32 = "int.smod";

// Remainder of division by zero is not defined.

let a: i32 = Mod(1, 0);

let b: i32 = Mod(0, 0);

// CHECK:STDOUT: --- int_div.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mod.type.1: type = fn_type @Mod.1 [template]
// CHECK:STDOUT:   %Mod: %Mod.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_5, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_5) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_3, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_3) [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [template]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type.1 = fn_decl @Mod.1 [template = constants.%Mod] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc2_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_19: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc2_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc2_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %Mod.ref: %Mod.type.1 = name_ref Mod, %Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %impl.elem0.loc7_20: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_20: <bound method> = bound_method %int_5, %impl.elem0.loc7_20 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_20: <specific function> = specific_function %Convert.bound.loc7_20, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc7_20: init %Int = call %Convert.specific_fn.loc7_20(%int_5) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_20.1: %i32 = value_of_initializer %int.convert_checked.loc7_20 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc7_20.2: %i32 = converted %int_5, %.loc7_20.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc7_23: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc7_23: <bound method> = bound_method %int_3, %impl.elem0.loc7_23 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc7_23: <specific function> = specific_function %Convert.bound.loc7_23, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc7_23: init %Int = call %Convert.specific_fn.loc7_23(%int_3) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_23.1: %i32 = value_of_initializer %int.convert_checked.loc7_23 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc7_23.2: %i32 = converted %int_3, %.loc7_23.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.smod: init %i32 = call %Mod.ref(%.loc7_20.2, %.loc7_23.2)
// CHECK:STDOUT:   %arr.var: ref <error> = var arr
// CHECK:STDOUT:   %arr: ref <error> = bind_name arr, %arr.var
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [template = constants.%array_type]
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
// CHECK:STDOUT: fn @Mod.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.smod: init %i32 = call %Mod.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %int.smod
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int.smod, %.loc11_19.1
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
// CHECK:STDOUT: --- fail_overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mod.type.1: type = fn_type @Mod.1 [template]
// CHECK:STDOUT:   %Mod: %Mod.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Sub.type.1: type = fn_type @Sub.1 [template]
// CHECK:STDOUT:   %Sub: %Sub.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Negate.type.1: type = fn_type @Negate.1 [template]
// CHECK:STDOUT:   %Negate: %Negate.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_2147483647, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_2147483647) [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_1) [symbolic]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type.1 = fn_decl @Mod.1 [template = constants.%Mod] {
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
// CHECK:STDOUT:   %Sub.decl: %Sub.type.1 = fn_decl @Sub.1 [template = constants.%Sub] {
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
// CHECK:STDOUT:   %Negate.decl: %Negate.type.1 = fn_decl @Negate.1 [template = constants.%Negate] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_14: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc6_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.ssub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate.1(%a.param_patt: %i32) -> %i32 = "int.snegate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref.loc9: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Negate.ref.loc9_18: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc9: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc9_25: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_25: <bound method> = bound_method %int_2147483647.loc9, %impl.elem0.loc9_25 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_25: <specific function> = specific_function %Convert.bound.loc9_25, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc9_25: init %Int = call %Convert.specific_fn.loc9_25(%int_2147483647.loc9) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_25.1: %i32 = value_of_initializer %int.convert_checked.loc9_25 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc9_25.2: %i32 = converted %int_2147483647.loc9, %.loc9_25.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate.loc9_36: init %i32 = call %Negate.ref.loc9_18(%.loc9_25.2)
// CHECK:STDOUT:   %Negate.ref.loc9_39: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc9_46: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc9_46: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_46 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc9_46: <specific function> = specific_function %Convert.bound.loc9_46, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc9_46: init %Int = call %Convert.specific_fn.loc9_46(%int_1.loc9) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_46.1: %i32 = value_of_initializer %int.convert_checked.loc9_46 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc9_46.2: %i32 = converted %int_1.loc9, %.loc9_46.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.snegate.loc9_47: init %i32 = call %Negate.ref.loc9_39(%.loc9_46.2)
// CHECK:STDOUT:   %.loc9_36.1: %i32 = value_of_initializer %int.snegate.loc9_36
// CHECK:STDOUT:   %.loc9_36.2: %i32 = converted %int.snegate.loc9_36, %.loc9_36.1
// CHECK:STDOUT:   %.loc9_47.1: %i32 = value_of_initializer %int.snegate.loc9_47
// CHECK:STDOUT:   %.loc9_47.2: %i32 = converted %int.snegate.loc9_47, %.loc9_47.1
// CHECK:STDOUT:   %int.smod.loc9: init %i32 = call %Mod.ref.loc9(%.loc9_36.2, %.loc9_47.2)
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer %int.smod.loc9
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted %int.smod.loc9, %.loc9_49.1
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc9_49.2
// CHECK:STDOUT:   %Mod.ref.loc12: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Sub.ref.loc12: %Sub.type.1 = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc12: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc12: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc12_29: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc12_29: <bound method> = bound_method %int_2147483647.loc12, %impl.elem0.loc12_29 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_29: <specific function> = specific_function %Convert.bound.loc12_29, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_29: init %Int = call %Convert.specific_fn.loc12_29(%int_2147483647.loc12) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc12_29.1: %i32 = value_of_initializer %int.convert_checked.loc12_29 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc12_29.2: %i32 = converted %int_2147483647.loc12, %.loc12_29.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate.loc12: init %i32 = call %Negate.ref.loc12(%.loc12_29.2)
// CHECK:STDOUT:   %int_1.loc12_43: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc12_40.1: %i32 = value_of_initializer %int.snegate.loc12
// CHECK:STDOUT:   %.loc12_40.2: %i32 = converted %int.snegate.loc12, %.loc12_40.1
// CHECK:STDOUT:   %impl.elem0.loc12_43: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc12_43: <bound method> = bound_method %int_1.loc12_43, %impl.elem0.loc12_43 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_43: <specific function> = specific_function %Convert.bound.loc12_43, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_43: init %Int = call %Convert.specific_fn.loc12_43(%int_1.loc12_43) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_43.1: %i32 = value_of_initializer %int.convert_checked.loc12_43 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_43.2: %i32 = converted %int_1.loc12_43, %.loc12_43.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.ssub.loc12: init %i32 = call %Sub.ref.loc12(%.loc12_40.2, %.loc12_43.2)
// CHECK:STDOUT:   %int_1.loc12_47: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc12_44.1: %i32 = value_of_initializer %int.ssub.loc12
// CHECK:STDOUT:   %.loc12_44.2: %i32 = converted %int.ssub.loc12, %.loc12_44.1
// CHECK:STDOUT:   %impl.elem0.loc12_47: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc12_47: <bound method> = bound_method %int_1.loc12_47, %impl.elem0.loc12_47 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_47: <specific function> = specific_function %Convert.bound.loc12_47, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_47: init %Int = call %Convert.specific_fn.loc12_47(%int_1.loc12_47) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_47.1: %i32 = value_of_initializer %int.convert_checked.loc12_47 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc12_47.2: %i32 = converted %int_1.loc12_47, %.loc12_47.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.smod.loc12: init %i32 = call %Mod.ref.loc12(%.loc12_44.2, %.loc12_47.2)
// CHECK:STDOUT:   %.loc12_49.1: %i32 = value_of_initializer %int.smod.loc12
// CHECK:STDOUT:   %.loc12_49.2: %i32 = converted %int.smod.loc12, %.loc12_49.1
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc12_49.2
// CHECK:STDOUT:   %Mod.ref.loc16: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %Sub.ref.loc16: %Sub.type.1 = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Negate.ref.loc16_22: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_2147483647.loc16: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647]
// CHECK:STDOUT:   %impl.elem0.loc16_29: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc16_29: <bound method> = bound_method %int_2147483647.loc16, %impl.elem0.loc16_29 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_29: <specific function> = specific_function %Convert.bound.loc16_29, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_29: init %Int = call %Convert.specific_fn.loc16_29(%int_2147483647.loc16) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc16_29.1: %i32 = value_of_initializer %int.convert_checked.loc16_29 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc16_29.2: %i32 = converted %int_2147483647.loc16, %.loc16_29.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %int.snegate.loc16_40: init %i32 = call %Negate.ref.loc16_22(%.loc16_29.2)
// CHECK:STDOUT:   %int_1.loc16_43: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc16_40.1: %i32 = value_of_initializer %int.snegate.loc16_40
// CHECK:STDOUT:   %.loc16_40.2: %i32 = converted %int.snegate.loc16_40, %.loc16_40.1
// CHECK:STDOUT:   %impl.elem0.loc16_43: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc16_43: <bound method> = bound_method %int_1.loc16_43, %impl.elem0.loc16_43 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_43: <specific function> = specific_function %Convert.bound.loc16_43, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc16_43: init %Int = call %Convert.specific_fn.loc16_43(%int_1.loc16_43) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc16_43.1: %i32 = value_of_initializer %int.convert_checked.loc16_43 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc16_43.2: %i32 = converted %int_1.loc16_43, %.loc16_43.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.ssub.loc16: init %i32 = call %Sub.ref.loc16(%.loc16_40.2, %.loc16_43.2)
// CHECK:STDOUT:   %Negate.ref.loc16_47: %Negate.type.1 = name_ref Negate, file.%Negate.decl [template = constants.%Negate]
// CHECK:STDOUT:   %int_1.loc16_54: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %impl.elem0.loc16_54: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc16_54: <bound method> = bound_method %int_1.loc16_54, %impl.elem0.loc16_54 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_54: <specific function> = specific_function %Convert.bound.loc16_54, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc16_54: init %Int = call %Convert.specific_fn.loc16_54(%int_1.loc16_54) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc16_54.1: %i32 = value_of_initializer %int.convert_checked.loc16_54 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc16_54.2: %i32 = converted %int_1.loc16_54, %.loc16_54.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.snegate.loc16_55: init %i32 = call %Negate.ref.loc16_47(%.loc16_54.2)
// CHECK:STDOUT:   %.loc16_44.1: %i32 = value_of_initializer %int.ssub.loc16
// CHECK:STDOUT:   %.loc16_44.2: %i32 = converted %int.ssub.loc16, %.loc16_44.1
// CHECK:STDOUT:   %.loc16_55.1: %i32 = value_of_initializer %int.snegate.loc16_55
// CHECK:STDOUT:   %.loc16_55.2: %i32 = converted %int.snegate.loc16_55, %.loc16_55.1
// CHECK:STDOUT:   %int.smod.loc16: init %i32 = call %Mod.ref.loc16(%.loc16_44.2, %.loc16_55.2)
// CHECK:STDOUT:   %.loc16_57.1: %i32 = value_of_initializer %int.smod.loc16
// CHECK:STDOUT:   %.loc16_57.2: %i32 = converted %int.smod.loc16, %.loc16_57.1
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc16_57.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_div_by_zero.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %Int: type = class_type @Int, @Int(%N) [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Mod.type.1: type = fn_type @Mod.1 [template]
// CHECK:STDOUT:   %Mod: %Mod.type.1 = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.2, @impl.1(%N) [symbolic]
// CHECK:STDOUT:   %Convert.3: %Convert.type.3 = struct_value () [symbolic]
// CHECK:STDOUT:   %impl_witness.5: <witness> = impl_witness (%Convert.3), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.1: init %Int = call %Convert.specific_fn.1(%int_1) [symbolic]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0, %Convert.3 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%N) [symbolic]
// CHECK:STDOUT:   %int.convert_checked.2: init %Int = call %Convert.specific_fn.2(%int_0) [symbolic]
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
// CHECK:STDOUT:     .Mod = %Mod.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Mod.decl: %Mod.type.1 = fn_decl @Mod.1 [template = constants.%Mod] {
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
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mod.1(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.smod";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mod.ref.loc8: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %int_0.loc8: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc8_18: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_18: <bound method> = bound_method %int_1, %impl.elem0.loc8_18 [symbolic = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_18: <specific function> = specific_function %Convert.bound.loc8_18, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc8_18: init %Int = call %Convert.specific_fn.loc8_18(%int_1) [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_18.1: %i32 = value_of_initializer %int.convert_checked.loc8_18 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %.loc8_18.2: %i32 = converted %int_1, %.loc8_18.1 [symbolic = constants.%int.convert_checked.1]
// CHECK:STDOUT:   %impl.elem0.loc8_21: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc8_21: <bound method> = bound_method %int_0.loc8, %impl.elem0.loc8_21 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc8_21: <specific function> = specific_function %Convert.bound.loc8_21, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc8_21: init %Int = call %Convert.specific_fn.loc8_21(%int_0.loc8) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_21.1: %i32 = value_of_initializer %int.convert_checked.loc8_21 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc8_21.2: %i32 = converted %int_0.loc8, %.loc8_21.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.smod.loc8: init %i32 = call %Mod.ref.loc8(%.loc8_18.2, %.loc8_21.2)
// CHECK:STDOUT:   %.loc8_23.1: %i32 = value_of_initializer %int.smod.loc8
// CHECK:STDOUT:   %.loc8_23.2: %i32 = converted %int.smod.loc8, %.loc8_23.1
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc8_23.2
// CHECK:STDOUT:   %Mod.ref.loc10: %Mod.type.1 = name_ref Mod, file.%Mod.decl [template = constants.%Mod]
// CHECK:STDOUT:   %int_0.loc10_18: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %int_0.loc10_21: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %impl.elem0.loc10_18: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc10_18: <bound method> = bound_method %int_0.loc10_18, %impl.elem0.loc10_18 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_18: <specific function> = specific_function %Convert.bound.loc10_18, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_18: init %Int = call %Convert.specific_fn.loc10_18(%int_0.loc10_18) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc10_18.1: %i32 = value_of_initializer %int.convert_checked.loc10_18 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc10_18.2: %i32 = converted %int_0.loc10_18, %.loc10_18.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %impl.elem0.loc10_21: %Convert.type.2 = impl_witness_access constants.%impl_witness.5, element0 [symbolic = constants.%Convert.3]
// CHECK:STDOUT:   %Convert.bound.loc10_21: <bound method> = bound_method %int_0.loc10_21, %impl.elem0.loc10_21 [symbolic = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc10_21: <specific function> = specific_function %Convert.bound.loc10_21, @Convert.2(constants.%N) [symbolic = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc10_21: init %Int = call %Convert.specific_fn.loc10_21(%int_0.loc10_21) [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc10_21.1: %i32 = value_of_initializer %int.convert_checked.loc10_21 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %.loc10_21.2: %i32 = converted %int_0.loc10_21, %.loc10_21.1 [symbolic = constants.%int.convert_checked.2]
// CHECK:STDOUT:   %int.smod.loc10: init %i32 = call %Mod.ref.loc10(%.loc10_18.2, %.loc10_21.2)
// CHECK:STDOUT:   %.loc10_23.1: %i32 = value_of_initializer %int.smod.loc10
// CHECK:STDOUT:   %.loc10_23.2: %i32 = converted %int.smod.loc10, %.loc10_23.1
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc10_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
