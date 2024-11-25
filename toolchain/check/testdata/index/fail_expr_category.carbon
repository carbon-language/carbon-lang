// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_expr_category.carbon

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  // Indexing an array value gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:18: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   var pb: i32* = &b[0];
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  var pb: i32* = &b[0];
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   b[0] = 4;
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  b[0] = 4;

  // Indexing an ephemeral reference (materialized from an initializing
  // expression) gives a value.
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+4]]:18: error: cannot take the address of non-reference expression [AddrOfNonRef]
  // CHECK:STDERR:   var pf: i32* = &F()[0];
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  var pf: i32* = &F()[0];
  // CHECK:STDERR: fail_expr_category.carbon:[[@LINE+3]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   F()[0] = 4;
  // CHECK:STDERR:   ^~~~~~
  F()[0] = 4;
}

// CHECK:STDOUT: --- fail_expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.34, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 4 [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.3 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_12.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_17: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_12.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_12.3: type = converted %int.make_type_signed, %.loc11_12.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17, %i32 [template = constants.%.3]
// CHECK:STDOUT:     %return.param: ref %.3 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %b.patt: %.3 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.3 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_15: Core.IntLiteral = int_value 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_10.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_10.3: type = converted %int.make_type_signed.loc13, %.loc13_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_16: type = array_type %.loc13_15, %i32 [template = constants.%.3]
// CHECK:STDOUT:     %b.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %b: %.3 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%.loc19_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.1: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.2: type = converted %int.make_type_signed.loc19, %.loc19_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.3: type = ptr_type %i32 [template = constants.%.5]
// CHECK:STDOUT:   %pb.var: ref %.5 = var pb
// CHECK:STDOUT:   %pb: ref %.5 = bind_name pb, %pb.var
// CHECK:STDOUT:   %b.ref.loc19: %.3 = name_ref b, %b
// CHECK:STDOUT:   %.loc19_21.1: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc19_21.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_21.3: <bound method> = bound_method %.loc19_21.1, %.loc19_21.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc19_21.4: <specific function> = specific_function %.loc19_21.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc19: init %i32 = call %.loc19_21.4(%.loc19_21.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc19_21.5: %i32 = value_of_initializer %int.convert_checked.loc19 [template = constants.%.33]
// CHECK:STDOUT:   %.loc19_21.6: %i32 = converted %.loc19_21.1, %.loc19_21.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc19_22.1: ref %.3 = value_as_ref %b.ref.loc19
// CHECK:STDOUT:   %.loc19_22.2: ref %i32 = array_index %.loc19_22.1, %.loc19_21.6
// CHECK:STDOUT:   %.loc19_22.3: %i32 = bind_value %.loc19_22.2
// CHECK:STDOUT:   %.loc19_18: %.5 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pb.var, %.loc19_18
// CHECK:STDOUT:   %b.ref.loc24: %.3 = name_ref b, %b
// CHECK:STDOUT:   %.loc24_5.1: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc24_5.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_5.3: <bound method> = bound_method %.loc24_5.1, %.loc24_5.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc24_5.4: <specific function> = specific_function %.loc24_5.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc24_5: init %i32 = call %.loc24_5.4(%.loc24_5.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc24_5.5: %i32 = value_of_initializer %int.convert_checked.loc24_5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc24_5.6: %i32 = converted %.loc24_5.1, %.loc24_5.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc24_6.1: ref %.3 = value_as_ref %b.ref.loc24
// CHECK:STDOUT:   %.loc24_6.2: ref %i32 = array_index %.loc24_6.1, %.loc24_5.6
// CHECK:STDOUT:   %.loc24_6.3: %i32 = bind_value %.loc24_6.2
// CHECK:STDOUT:   %.loc24_10: Core.IntLiteral = int_value 4 [template = constants.%.34]
// CHECK:STDOUT:   %.loc24_8.1: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_8.2: <bound method> = bound_method %.loc24_10, %.loc24_8.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc24_8.3: <specific function> = specific_function %.loc24_8.2, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc24_8: init %i32 = call %.loc24_8.3(%.loc24_10) [template = constants.%.37]
// CHECK:STDOUT:   %.loc24_8.4: init %i32 = converted %.loc24_10, %int.convert_checked.loc24_8 [template = constants.%.37]
// CHECK:STDOUT:   assign %.loc24_6.3, %.loc24_8.4
// CHECK:STDOUT:   %.loc32_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc32: init type = call constants.%Int(%.loc32_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.1: type = value_of_initializer %int.make_type_signed.loc32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.2: type = converted %int.make_type_signed.loc32, %.loc32_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc32_14.3: type = ptr_type %i32 [template = constants.%.5]
// CHECK:STDOUT:   %pf.var: ref %.5 = var pf
// CHECK:STDOUT:   %pf: ref %.5 = bind_name pf, %pf.var
// CHECK:STDOUT:   %F.ref.loc32: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc32_20.1: ref %.3 = temporary_storage
// CHECK:STDOUT:   %F.call.loc32: init %.3 = call %F.ref.loc32() to %.loc32_20.1
// CHECK:STDOUT:   %.loc32_23.1: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc32_20.2: ref %.3 = temporary %.loc32_20.1, %F.call.loc32
// CHECK:STDOUT:   %.loc32_23.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc32_23.3: <bound method> = bound_method %.loc32_23.1, %.loc32_23.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc32_23.4: <specific function> = specific_function %.loc32_23.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc32: init %i32 = call %.loc32_23.4(%.loc32_23.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc32_23.5: %i32 = value_of_initializer %int.convert_checked.loc32 [template = constants.%.33]
// CHECK:STDOUT:   %.loc32_23.6: %i32 = converted %.loc32_23.1, %.loc32_23.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc32_24.1: ref %i32 = array_index %.loc32_20.2, %.loc32_23.6
// CHECK:STDOUT:   %.loc32_24.2: %i32 = bind_value %.loc32_24.1
// CHECK:STDOUT:   %.loc32_18: %.5 = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign %pf.var, %.loc32_18
// CHECK:STDOUT:   %F.ref.loc36: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc36_4.1: ref %.3 = temporary_storage
// CHECK:STDOUT:   %F.call.loc36: init %.3 = call %F.ref.loc36() to %.loc36_4.1
// CHECK:STDOUT:   %.loc36_7.1: Core.IntLiteral = int_value 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc36_4.2: ref %.3 = temporary %.loc36_4.1, %F.call.loc36
// CHECK:STDOUT:   %.loc36_7.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc36_7.3: <bound method> = bound_method %.loc36_7.1, %.loc36_7.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc36_7.4: <specific function> = specific_function %.loc36_7.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc36_7: init %i32 = call %.loc36_7.4(%.loc36_7.1) [template = constants.%.33]
// CHECK:STDOUT:   %.loc36_7.5: %i32 = value_of_initializer %int.convert_checked.loc36_7 [template = constants.%.33]
// CHECK:STDOUT:   %.loc36_7.6: %i32 = converted %.loc36_7.1, %.loc36_7.5 [template = constants.%.33]
// CHECK:STDOUT:   %.loc36_8.1: ref %i32 = array_index %.loc36_4.2, %.loc36_7.6
// CHECK:STDOUT:   %.loc36_8.2: %i32 = bind_value %.loc36_8.1
// CHECK:STDOUT:   %.loc36_12: Core.IntLiteral = int_value 4 [template = constants.%.34]
// CHECK:STDOUT:   %.loc36_10.1: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc36_10.2: <bound method> = bound_method %.loc36_12, %.loc36_10.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc36_10.3: <specific function> = specific_function %.loc36_10.2, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc36_10: init %i32 = call %.loc36_10.3(%.loc36_12) [template = constants.%.37]
// CHECK:STDOUT:   %.loc36_10.4: init %i32 = converted %.loc36_12, %int.convert_checked.loc36_10 [template = constants.%.37]
// CHECK:STDOUT:   assign %.loc36_8.2, %.loc36_10.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
