// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/returned_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/returned_var.carbon

class C {
  var a: i32;
  var b: i32;
}

fn F() -> C {
  returned var result: C = {.a = 1, .b = 2};
  return var;
}

fn G() -> i32 {
  returned var result: i32 = 0;
  return var;
}

// CHECK:STDOUT: --- returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, %i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%.35, %.38) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.39, %Convert.14 [template]
// CHECK:STDOUT:   %.41: <specific function> = specific_function %.40, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.42: %i32 = int_value 0 [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc16: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc21: init type = call constants.%Int(%.loc21_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_11.2: type = value_of_initializer %int.make_type_signed.loc21 [template = constants.%i32]
// CHECK:STDOUT:     %.loc21_11.3: type = converted %int.make_type_signed.loc21, %.loc21_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed.loc12, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8: %.2 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc13_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_10.3: type = converted %int.make_type_signed.loc13, %.loc13_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8: %.2 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc14: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT:   .b = %.loc13_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc17: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %result: ref %C = bind_name result, %return
// CHECK:STDOUT:   %.loc17_34: Core.IntLiteral = int_value 1 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_42: Core.IntLiteral = int_value 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc17_43.1: %.8 = struct_literal (%.loc17_34, %.loc17_42)
// CHECK:STDOUT:   %.loc17_43.2: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_43.3: <bound method> = bound_method %.loc17_34, %.loc17_43.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc17_43.4: <specific function> = specific_function %.loc17_43.3, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc17_43.1: init %i32 = call %.loc17_43.4(%.loc17_34) [template = constants.%.35]
// CHECK:STDOUT:   %.loc17_43.5: init %i32 = converted %.loc17_34, %int.convert_checked.loc17_43.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc17_43.6: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc17_43.7: init %i32 = initialize_from %.loc17_43.5 to %.loc17_43.6 [template = constants.%.35]
// CHECK:STDOUT:   %.loc17_43.8: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_43.9: <bound method> = bound_method %.loc17_42, %.loc17_43.8 [template = constants.%.36]
// CHECK:STDOUT:   %.loc17_43.10: <specific function> = specific_function %.loc17_43.9, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc17_43.2: init %i32 = call %.loc17_43.10(%.loc17_42) [template = constants.%.38]
// CHECK:STDOUT:   %.loc17_43.11: init %i32 = converted %.loc17_42, %int.convert_checked.loc17_43.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc17_43.12: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc17_43.13: init %i32 = initialize_from %.loc17_43.11 to %.loc17_43.12 [template = constants.%.38]
// CHECK:STDOUT:   %.loc17_43.14: init %C = class_init (%.loc17_43.7, %.loc17_43.13), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc17_44: init %C = converted %.loc17_43.1, %.loc17_43.14 [template = constants.%struct]
// CHECK:STDOUT:   assign %return, %.loc17_44
// CHECK:STDOUT:   return %result to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_24.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%.loc22_24.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_24.2: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_24.3: type = converted %int.make_type_signed.loc22, %.loc22_24.2 [template = constants.%i32]
// CHECK:STDOUT:   %result.var: ref %i32 = var result
// CHECK:STDOUT:   %result: ref %i32 = bind_name result, %result.var
// CHECK:STDOUT:   %.loc22_30: Core.IntLiteral = int_value 0 [template = constants.%.39]
// CHECK:STDOUT:   %.loc22_31.1: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc22_31.2: <bound method> = bound_method %.loc22_30, %.loc22_31.1 [template = constants.%.40]
// CHECK:STDOUT:   %.loc22_31.3: <specific function> = specific_function %.loc22_31.2, @Convert.2(constants.%.1) [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc22_31.3(%.loc22_30) [template = constants.%.42]
// CHECK:STDOUT:   %.loc22_31.4: init %i32 = converted %.loc22_30, %int.convert_checked [template = constants.%.42]
// CHECK:STDOUT:   assign %result.var, %.loc22_31.4
// CHECK:STDOUT:   %.loc22_16: %i32 = bind_value %result
// CHECK:STDOUT:   return %.loc22_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
