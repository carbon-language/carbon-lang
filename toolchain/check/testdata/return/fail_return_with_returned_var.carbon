// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_return_with_returned_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_return_with_returned_var.carbon

fn F() -> i32 {
  returned var v: i32 = 0;
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+7]]:3: error: can only `return var;` in the scope of a `returned var` [ReturnExprWithReturnedVar]
  // CHECK:STDERR:   return 1;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: note: `returned var` was declared here [ReturnedVarHere]
  // CHECK:STDERR:   returned var v: i32 = 0;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  return 1;
}

class C { var a: i32; var b: i32; }
fn G() -> C {
  returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE+6]]:3: error: can only `return var;` in the scope of a `returned var` [ReturnExprWithReturnedVar]
  // CHECK:STDERR:   return c;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_return_with_returned_var.carbon:[[@LINE-4]]:16: note: `returned var` was declared here [ReturnedVarHere]
  // CHECK:STDERR:   returned var c: C = {.a = 1, .b = 2};
  // CHECK:STDERR:                ^
  return c;
}

// CHECK:STDOUT: --- fail_return_with_returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%.2) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.27: <specific function> = specific_function %.26, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.28: %.2 = int_value 0 [template]
// CHECK:STDOUT:   %.29: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.30: type = unbound_element_type %C, %.2 [template]
// CHECK:STDOUT:   %.31: type = struct_type {.a: %.2, .b: %.2} [template]
// CHECK:STDOUT:   %.32: <witness> = complete_type_witness %.31 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.35: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.29, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %.2 = int_value 1 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.34, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.41: %.2 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %C = struct_value (%.38, %.41) [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11: init type = call constants.%Int(%.loc11_11.1) [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_11.2: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_11.3: type = converted %int.make_type_signed.loc11, %.loc11_11.2 [template = constants.%.2]
// CHECK:STDOUT:     %return.param: ref %.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc23_18.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc23_18: init type = call constants.%Int(%.loc23_18.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_18.2: type = value_of_initializer %int.make_type_signed.loc23_18 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_18.3: type = converted %int.make_type_signed.loc23_18, %.loc23_18.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_16: %.30 = field_decl a, element0 [template]
// CHECK:STDOUT:   %.loc23_30.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc23_30: init type = call constants.%Int(%.loc23_30.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_30.2: type = value_of_initializer %int.make_type_signed.loc23_30 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_30.3: type = converted %int.make_type_signed.loc23_30, %.loc23_30.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_28: %.30 = field_decl b, element1 [template]
// CHECK:STDOUT:   %.loc23_35: <witness> = complete_type_witness %.31 [template = constants.%.32]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .a = %.loc23_16
// CHECK:STDOUT:   .b = %.loc23_28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_19.1) [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_19.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_19.3: type = converted %int.make_type_signed.loc12, %.loc12_19.2 [template = constants.%.2]
// CHECK:STDOUT:   %v.var: ref %.2 = var v
// CHECK:STDOUT:   %v: ref %.2 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc12_25: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_26.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_26.2: <bound method> = bound_method %.loc12_25, %.loc12_26.1 [template = constants.%.26]
// CHECK:STDOUT:   %.loc12_26.3: <specific function> = specific_function %.loc12_26.2, @Convert.2(constants.%.1) [template = constants.%.27]
// CHECK:STDOUT:   %int.convert_checked: init %.2 = call %.loc12_26.3(%.loc12_25) [template = constants.%.28]
// CHECK:STDOUT:   %.loc12_26.4: init %.2 = converted %.loc12_25, %int.convert_checked [template = constants.%.28]
// CHECK:STDOUT:   assign %v.var, %.loc12_26.4
// CHECK:STDOUT:   %.loc20: Core.IntLiteral = int_value 1 [template = constants.%.29]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref.loc25: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %return
// CHECK:STDOUT:   %.loc25_29: Core.IntLiteral = int_value 1 [template = constants.%.29]
// CHECK:STDOUT:   %.loc25_37: Core.IntLiteral = int_value 2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc25_38.1: %.35 = struct_literal (%.loc25_29, %.loc25_37)
// CHECK:STDOUT:   %.loc25_38.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc25_38.3: <bound method> = bound_method %.loc25_29, %.loc25_38.2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc25_38.4: <specific function> = specific_function %.loc25_38.3, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc25_38.1: init %.2 = call %.loc25_38.4(%.loc25_29) [template = constants.%.38]
// CHECK:STDOUT:   %.loc25_38.5: init %.2 = converted %.loc25_29, %int.convert_checked.loc25_38.1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc25_38.6: ref %.2 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc25_38.7: init %.2 = initialize_from %.loc25_38.5 to %.loc25_38.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc25_38.8: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc25_38.9: <bound method> = bound_method %.loc25_37, %.loc25_38.8 [template = constants.%.39]
// CHECK:STDOUT:   %.loc25_38.10: <specific function> = specific_function %.loc25_38.9, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc25_38.2: init %.2 = call %.loc25_38.10(%.loc25_37) [template = constants.%.41]
// CHECK:STDOUT:   %.loc25_38.11: init %.2 = converted %.loc25_37, %int.convert_checked.loc25_38.2 [template = constants.%.41]
// CHECK:STDOUT:   %.loc25_38.12: ref %.2 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc25_38.13: init %.2 = initialize_from %.loc25_38.11 to %.loc25_38.12 [template = constants.%.41]
// CHECK:STDOUT:   %.loc25_38.14: init %C = class_init (%.loc25_38.7, %.loc25_38.13), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc25_39: init %C = converted %.loc25_38.1, %.loc25_38.14 [template = constants.%struct]
// CHECK:STDOUT:   assign %return, %.loc25_39
// CHECK:STDOUT:   %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
