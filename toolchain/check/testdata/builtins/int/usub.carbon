// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/usub.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/usub.carbon

// --- int_sub.carbon

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

var arr: [i32; Sub(3, 2)];
let arr_p: [i32; 1]* = &arr;

fn RuntimeCall(a: i32, b: i32) -> i32 {
  return Sub(a, b);
}

// --- overflow.carbon

package Overflow;

fn Sub(a: i32, b: i32) -> i32 = "int.usub";

let a: i32 = Sub(0, 0x7FFFFFFF);
let b: i32 = Sub(Sub(0, 0x7FFFFFFF), 1);
let c: i32 = Sub(Sub(0, 0x7FFFFFFF), 2);

// CHECK:STDOUT: --- int_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %int.3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.16: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %.17: <specific function> = specific_function %.16, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.19: <specific function> = specific_function %.18, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %int.1.1: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.4, @impl.3(%int.32) [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.20: <bound method> = bound_method %int.1.1, %Convert.15 [template]
// CHECK:STDOUT:   %.21: <specific function> = specific_function %.20, @Convert.4(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.22: type = array_type %int.1.2, %i32 [template]
// CHECK:STDOUT:   %.23: type = ptr_type %.22 [template]
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
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:     .arr_p = @__global_init.%arr_p
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc2_11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_11: init type = call constants.%Int(%int.32.loc2_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.1: type = value_of_initializer %int.make_type_signed.loc2_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_11.2: type = converted %int.make_type_signed.loc2_11, %.loc2_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc2_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_19: init type = call constants.%Int(%int.32.loc2_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.1: type = value_of_initializer %int.make_type_signed.loc2_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_19.2: type = converted %int.make_type_signed.loc2_19, %.loc2_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc2_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc2_27: init type = call constants.%Int(%int.32.loc2_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %int.make_type_signed.loc2_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %int.make_type_signed.loc2_27, %.loc2_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc4: init type = call constants.%Int(%int.32.loc4) [template = constants.%i32]
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, %Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc4_20.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_20.2: <bound method> = bound_method %int.3, %.loc4_20.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc4_20.3: <specific function> = specific_function %.loc4_20.2, @Convert.2(constants.%int.32) [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked.loc4_20: init %i32 = call %.loc4_20.3(%int.3) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc4_20.4: %i32 = value_of_initializer %int.convert_checked.loc4_20 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc4_20.5: %i32 = converted %int.3, %.loc4_20.4 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc4_23.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc4_23.2: <bound method> = bound_method %int.2, %.loc4_23.1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc4_23.3: <specific function> = specific_function %.loc4_23.2, @Convert.2(constants.%int.32) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc4_23: init %i32 = call %.loc4_23.3(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc4_23.4: %i32 = value_of_initializer %int.convert_checked.loc4_23 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc4_23.5: %i32 = converted %int.2, %.loc4_23.4 [template = constants.%int.2.2]
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%.loc4_20.5, %.loc4_23.5) [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_11.2: type = converted %int.make_type_signed.loc4, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc4_24.1: %Convert.type.6 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc4_24.2: <bound method> = bound_method %int.usub, %.loc4_24.1 [template = constants.%.20]
// CHECK:STDOUT:   %.loc4_24.3: <specific function> = specific_function %.loc4_24.2, @Convert.4(constants.%int.32) [template = constants.%.21]
// CHECK:STDOUT:   %.loc4_24.4: %i32 = value_of_initializer %int.usub [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc4_24.5: %i32 = converted %int.usub, %.loc4_24.4 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.convert_checked.loc4_24: init Core.IntLiteral = call %.loc4_24.3(%.loc4_24.5) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc4_24.6: Core.IntLiteral = value_of_initializer %int.convert_checked.loc4_24 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc4_24.7: Core.IntLiteral = converted %int.usub, %.loc4_24.6 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc4_25: type = array_type %.loc4_24.7, %i32 [template = constants.%.22]
// CHECK:STDOUT:   %arr.var: ref %.22 = var arr
// CHECK:STDOUT:   %arr: ref %.22 = bind_name arr, %arr.var
// CHECK:STDOUT:   %int.32.loc5: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%int.32.loc5) [template = constants.%i32]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc5_13.1: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_13.2: type = converted %int.make_type_signed.loc5, %.loc5_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_19: type = array_type %int.1, %i32 [template = constants.%.22]
// CHECK:STDOUT:   %.loc5_20: type = ptr_type %.22 [template = constants.%.23]
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc7_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_19: init type = call constants.%Int(%int.32.loc7_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.1: type = value_of_initializer %int.make_type_signed.loc7_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_19.2: type = converted %int.make_type_signed.loc7_19, %.loc7_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc7_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_27: init type = call constants.%Int(%int.32.loc7_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.1: type = value_of_initializer %int.make_type_signed.loc7_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_27.2: type = converted %int.make_type_signed.loc7_27, %.loc7_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc7_35: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_35: init type = call constants.%Int(%int.32.loc7_35) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.1: type = value_of_initializer %int.make_type_signed.loc7_35 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_35.2: type = converted %int.make_type_signed.loc7_35, %.loc7_35.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: %i32 = name_ref b, %b
// CHECK:STDOUT:   %int.usub: init %i32 = call %Sub.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc8_19.1: %i32 = value_of_initializer %int.usub
// CHECK:STDOUT:   %.loc8_19.2: %i32 = converted %int.usub, %.loc8_19.1
// CHECK:STDOUT:   return %.loc8_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: ref %.22 = name_ref arr, file.%arr
// CHECK:STDOUT:   %.loc5: %.23 = addr_of %arr.ref
// CHECK:STDOUT:   %arr_p: %.23 = bind_name arr_p, %.loc5
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- overflow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %Sub.type: type = fn_type @Sub [template]
// CHECK:STDOUT:   %Sub: %Sub.type = struct_value () [template]
// CHECK:STDOUT:   %int.0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int.2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.16: <bound method> = bound_method %int.0.1, %Convert.14 [template]
// CHECK:STDOUT:   %.17: <specific function> = specific_function %.16, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.2147483647.1, %Convert.14 [template]
// CHECK:STDOUT:   %.19: <specific function> = specific_function %.18, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int.-2147483647: %i32 = int_value -2147483647 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.20: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.21: <specific function> = specific_function %.20, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int.-2147483648: %i32 = int_value -2147483648 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.22: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.23: <specific function> = specific_function %.22, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .a = @__global_init.%a
// CHECK:STDOUT:     .b = @__global_init.%b
// CHECK:STDOUT:     .c = @__global_init.%c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Sub.decl: %Sub.type = fn_decl @Sub [template = constants.%Sub] {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %i32 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.32.loc4_11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_11: init type = call constants.%Int(%int.32.loc4_11) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.1: type = value_of_initializer %int.make_type_signed.loc4_11 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_11.2: type = converted %int.make_type_signed.loc4_11, %.loc4_11.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc4_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_19: init type = call constants.%Int(%int.32.loc4_19) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.1: type = value_of_initializer %int.make_type_signed.loc4_19 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_19.2: type = converted %int.make_type_signed.loc4_19, %.loc4_19.1 [template = constants.%i32]
// CHECK:STDOUT:     %int.32.loc4_27: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:     %int.make_type_signed.loc4_27: init type = call constants.%Int(%int.32.loc4_27) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.1: type = value_of_initializer %int.make_type_signed.loc4_27 [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_27.2: type = converted %int.make_type_signed.loc4_27, %.loc4_27.1 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param1
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.32.loc6: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%int.32.loc6) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.1: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_8.2: type = converted %int.make_type_signed.loc6, %.loc6_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int.32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_signed.loc7, %.loc7_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int.32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_8.2: type = converted %int.make_type_signed.loc8, %.loc8_8.1 [template = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Sub(%a.param_patt: %i32, %b.param_patt: %i32) -> %i32 = "int.usub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref.loc6: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int.0.loc6: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %int.2147483647.loc6: Core.IntLiteral = int_value 2147483647 [template = constants.%int.2147483647.1]
// CHECK:STDOUT:   %.loc6_18.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_18.2: <bound method> = bound_method %int.0.loc6, %.loc6_18.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc6_18.3: <specific function> = specific_function %.loc6_18.2, @Convert.2(constants.%int.32) [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked.loc6_18: init %i32 = call %.loc6_18.3(%int.0.loc6) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc6_18.4: %i32 = value_of_initializer %int.convert_checked.loc6_18 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc6_18.5: %i32 = converted %int.0.loc6, %.loc6_18.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc6_21.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc6_21.2: <bound method> = bound_method %int.2147483647.loc6, %.loc6_21.1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc6_21.3: <specific function> = specific_function %.loc6_21.2, @Convert.2(constants.%int.32) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc6_21: init %i32 = call %.loc6_21.3(%int.2147483647.loc6) [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc6_21.4: %i32 = value_of_initializer %int.convert_checked.loc6_21 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc6_21.5: %i32 = converted %int.2147483647.loc6, %.loc6_21.4 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %int.usub.loc6: init %i32 = call %Sub.ref.loc6(%.loc6_18.5, %.loc6_21.5) [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %.loc6_32.1: %i32 = value_of_initializer %int.usub.loc6 [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %.loc6_32.2: %i32 = converted %int.usub.loc6, %.loc6_32.1 [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc6_32.2
// CHECK:STDOUT:   %Sub.ref.loc7_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc7_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int.0.loc7: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %int.2147483647.loc7: Core.IntLiteral = int_value 2147483647 [template = constants.%int.2147483647.1]
// CHECK:STDOUT:   %.loc7_22.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_22.2: <bound method> = bound_method %int.0.loc7, %.loc7_22.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc7_22.3: <specific function> = specific_function %.loc7_22.2, @Convert.2(constants.%int.32) [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked.loc7_22: init %i32 = call %.loc7_22.3(%int.0.loc7) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc7_22.4: %i32 = value_of_initializer %int.convert_checked.loc7_22 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc7_22.5: %i32 = converted %int.0.loc7, %.loc7_22.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc7_25.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_25.2: <bound method> = bound_method %int.2147483647.loc7, %.loc7_25.1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc7_25.3: <specific function> = specific_function %.loc7_25.2, @Convert.2(constants.%int.32) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc7_25: init %i32 = call %.loc7_25.3(%int.2147483647.loc7) [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc7_25.4: %i32 = value_of_initializer %int.convert_checked.loc7_25 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc7_25.5: %i32 = converted %int.2147483647.loc7, %.loc7_25.4 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %int.usub.loc7_35: init %i32 = call %Sub.ref.loc7_18(%.loc7_22.5, %.loc7_25.5) [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc7_35.1: %i32 = value_of_initializer %int.usub.loc7_35 [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %.loc7_35.2: %i32 = converted %int.usub.loc7_35, %.loc7_35.1 [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %.loc7_38.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc7_38.2: <bound method> = bound_method %int.1, %.loc7_38.1 [template = constants.%.20]
// CHECK:STDOUT:   %.loc7_38.3: <specific function> = specific_function %.loc7_38.2, @Convert.2(constants.%int.32) [template = constants.%.21]
// CHECK:STDOUT:   %int.convert_checked.loc7_38: init %i32 = call %.loc7_38.3(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc7_38.4: %i32 = value_of_initializer %int.convert_checked.loc7_38 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc7_38.5: %i32 = converted %int.1, %.loc7_38.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %int.usub.loc7_39: init %i32 = call %Sub.ref.loc7_14(%.loc7_35.2, %.loc7_38.5) [template = constants.%int.-2147483648]
// CHECK:STDOUT:   %.loc7_40.1: %i32 = value_of_initializer %int.usub.loc7_39 [template = constants.%int.-2147483648]
// CHECK:STDOUT:   %.loc7_40.2: %i32 = converted %int.usub.loc7_39, %.loc7_40.1 [template = constants.%int.-2147483648]
// CHECK:STDOUT:   %b: %i32 = bind_name b, %.loc7_40.2
// CHECK:STDOUT:   %Sub.ref.loc8_14: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %Sub.ref.loc8_18: %Sub.type = name_ref Sub, file.%Sub.decl [template = constants.%Sub]
// CHECK:STDOUT:   %int.0.loc8: Core.IntLiteral = int_value 0 [template = constants.%int.0.1]
// CHECK:STDOUT:   %int.2147483647.loc8: Core.IntLiteral = int_value 2147483647 [template = constants.%int.2147483647.1]
// CHECK:STDOUT:   %.loc8_22.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_22.2: <bound method> = bound_method %int.0.loc8, %.loc8_22.1 [template = constants.%.16]
// CHECK:STDOUT:   %.loc8_22.3: <specific function> = specific_function %.loc8_22.2, @Convert.2(constants.%int.32) [template = constants.%.17]
// CHECK:STDOUT:   %int.convert_checked.loc8_22: init %i32 = call %.loc8_22.3(%int.0.loc8) [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc8_22.4: %i32 = value_of_initializer %int.convert_checked.loc8_22 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc8_22.5: %i32 = converted %int.0.loc8, %.loc8_22.4 [template = constants.%int.0.2]
// CHECK:STDOUT:   %.loc8_25.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_25.2: <bound method> = bound_method %int.2147483647.loc8, %.loc8_25.1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc8_25.3: <specific function> = specific_function %.loc8_25.2, @Convert.2(constants.%int.32) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc8_25: init %i32 = call %.loc8_25.3(%int.2147483647.loc8) [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc8_25.4: %i32 = value_of_initializer %int.convert_checked.loc8_25 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc8_25.5: %i32 = converted %int.2147483647.loc8, %.loc8_25.4 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %int.usub.loc8_35: init %i32 = call %Sub.ref.loc8_18(%.loc8_22.5, %.loc8_25.5) [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc8_35.1: %i32 = value_of_initializer %int.usub.loc8_35 [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %.loc8_35.2: %i32 = converted %int.usub.loc8_35, %.loc8_35.1 [template = constants.%int.-2147483647]
// CHECK:STDOUT:   %.loc8_38.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc8_38.2: <bound method> = bound_method %int.2, %.loc8_38.1 [template = constants.%.22]
// CHECK:STDOUT:   %.loc8_38.3: <specific function> = specific_function %.loc8_38.2, @Convert.2(constants.%int.32) [template = constants.%.23]
// CHECK:STDOUT:   %int.convert_checked.loc8_38: init %i32 = call %.loc8_38.3(%int.2) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc8_38.4: %i32 = value_of_initializer %int.convert_checked.loc8_38 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc8_38.5: %i32 = converted %int.2, %.loc8_38.4 [template = constants.%int.2.2]
// CHECK:STDOUT:   %int.usub.loc8_39: init %i32 = call %Sub.ref.loc8_14(%.loc8_35.2, %.loc8_38.5) [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc8_40.1: %i32 = value_of_initializer %int.usub.loc8_39 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc8_40.2: %i32 = converted %int.usub.loc8_39, %.loc8_40.1 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %c: %i32 = bind_name c, %.loc8_40.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
