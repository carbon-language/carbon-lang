// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_non_assignable.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_non_assignable.carbon

fn F() -> i32;

fn Main() {
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  1 = 2;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  F() = 1;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  i32 = i32*;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  (if true then a else a) = 10;
}

// CHECK:STDOUT: --- fail_assignment_to_non_assignable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.29: <specific function> = specific_function %.28, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.30: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.31: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.1 = tuple_value (%.31, %.32) [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.1 = tuple_value (%.2, %.3) [template]
// CHECK:STDOUT:   %.34: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.34, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.37: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.3: %tuple.type.2 = tuple_value (%.30, %.41) [template]
// CHECK:STDOUT:   %.42: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.43: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.1: %.43 = struct_value (%.31, %.32) [template]
// CHECK:STDOUT:   %struct.2: %.43 = struct_value (%.2, %.3) [template]
// CHECK:STDOUT:   %.45: bool = bool_literal true [template]
// CHECK:STDOUT:   %.46: <bound method> = bound_method %.31, %Convert.14 [template]
// CHECK:STDOUT:   %.47: <specific function> = specific_function %.46, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.48: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.49: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %.50: <bound method> = bound_method %.49, %Convert.14 [template]
// CHECK:STDOUT:   %.51: <specific function> = specific_function %.50, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.52: %i32 = int_value 10 [template]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.3: type = converted %int.make_type_signed, %.loc11_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_3: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_7: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   assign %.loc18_3, %.loc18_7
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %.loc23_9: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc23_7.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc23_7.2: <bound method> = bound_method %.loc23_9, %.loc23_7.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc23_7.3: <specific function> = specific_function %.loc23_7.2, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %.loc23_7.3(%.loc23_9) [template = constants.%.30]
// CHECK:STDOUT:   %.loc23_7.4: init %i32 = converted %.loc23_9, %int.convert_checked.loc23 [template = constants.%.30]
// CHECK:STDOUT:   assign %F.call, %.loc23_7.4
// CHECK:STDOUT:   %.loc28_4: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc28_7: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc28_8.1: %tuple.type.1 = tuple_literal (%.loc28_4, %.loc28_7)
// CHECK:STDOUT:   %.loc28_13: Core.IntLiteral = int_value 3 [template = constants.%.31]
// CHECK:STDOUT:   %.loc28_16: Core.IntLiteral = int_value 4 [template = constants.%.32]
// CHECK:STDOUT:   %.loc28_17.1: %tuple.type.1 = tuple_literal (%.loc28_13, %.loc28_16)
// CHECK:STDOUT:   %.loc28_17.2: Core.IntLiteral = tuple_access %.loc28_8.1, element0
// CHECK:STDOUT:   %.loc28_17.3: init Core.IntLiteral = initialize_from %.loc28_13 to %.loc28_17.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc28_17.4: Core.IntLiteral = tuple_access %.loc28_8.1, element1
// CHECK:STDOUT:   %.loc28_17.5: init Core.IntLiteral = initialize_from %.loc28_16 to %.loc28_17.4 [template = constants.%.32]
// CHECK:STDOUT:   %.loc28_17.6: init %tuple.type.1 = tuple_init (%.loc28_17.3, %.loc28_17.5) to %.loc28_8.1 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc28_10: init %tuple.type.1 = converted %.loc28_17.1, %.loc28_17.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign %.loc28_8.1, %.loc28_10
// CHECK:STDOUT:   %tuple.loc28: %tuple.type.1 = tuple_value (%.loc28_4, %.loc28_7) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc28_8.2: %tuple.type.1 = converted %.loc28_8.1, %tuple.loc28 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc29_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%.loc29_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_10.2: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_10.3: type = converted %int.make_type_signed.loc29, %.loc29_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %n.var: ref %i32 = var n
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   %.loc29_16: Core.IntLiteral = int_value 0 [template = constants.%.34]
// CHECK:STDOUT:   %.loc29_17.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc29_17.2: <bound method> = bound_method %.loc29_16, %.loc29_17.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc29_17.3: <specific function> = specific_function %.loc29_17.2, @Convert.2(constants.%.1) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %i32 = call %.loc29_17.3(%.loc29_16) [template = constants.%.37]
// CHECK:STDOUT:   %.loc29_17.4: init %i32 = converted %.loc29_16, %int.convert_checked.loc29 [template = constants.%.37]
// CHECK:STDOUT:   assign %n.var, %.loc29_17.4
// CHECK:STDOUT:   %n.ref.loc34_4: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc34_7: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc34_8.1: %tuple.type.2 = tuple_literal (%n.ref.loc34_4, %n.ref.loc34_7)
// CHECK:STDOUT:   %.loc34_13: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_16: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc34_17.1: %tuple.type.1 = tuple_literal (%.loc34_13, %.loc34_16)
// CHECK:STDOUT:   %.loc34_17.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc34_17.3: <bound method> = bound_method %.loc34_13, %.loc34_17.2 [template = constants.%.28]
// CHECK:STDOUT:   %.loc34_17.4: <specific function> = specific_function %.loc34_17.3, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc34_17.1: init %i32 = call %.loc34_17.4(%.loc34_13) [template = constants.%.30]
// CHECK:STDOUT:   %.loc34_17.5: init %i32 = converted %.loc34_13, %int.convert_checked.loc34_17.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc34_17.6: %i32 = tuple_access %.loc34_8.1, element0
// CHECK:STDOUT:   %.loc34_17.7: init %i32 = initialize_from %.loc34_17.5 to %.loc34_17.6 [template = constants.%.30]
// CHECK:STDOUT:   %.loc34_17.8: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc34_17.9: <bound method> = bound_method %.loc34_16, %.loc34_17.8 [template = constants.%.39]
// CHECK:STDOUT:   %.loc34_17.10: <specific function> = specific_function %.loc34_17.9, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc34_17.2: init %i32 = call %.loc34_17.10(%.loc34_16) [template = constants.%.41]
// CHECK:STDOUT:   %.loc34_17.11: init %i32 = converted %.loc34_16, %int.convert_checked.loc34_17.2 [template = constants.%.41]
// CHECK:STDOUT:   %.loc34_17.12: %i32 = tuple_access %.loc34_8.1, element1
// CHECK:STDOUT:   %.loc34_17.13: init %i32 = initialize_from %.loc34_17.11 to %.loc34_17.12 [template = constants.%.41]
// CHECK:STDOUT:   %.loc34_17.14: init %tuple.type.2 = tuple_init (%.loc34_17.7, %.loc34_17.13) to %.loc34_8.1 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc34_10: init %tuple.type.2 = converted %.loc34_17.1, %.loc34_17.14 [template = constants.%tuple.3]
// CHECK:STDOUT:   assign %.loc34_8.1, %.loc34_10
// CHECK:STDOUT:   %.loc34_4: %i32 = bind_value %n.ref.loc34_4
// CHECK:STDOUT:   %.loc34_7: %i32 = bind_value %n.ref.loc34_7
// CHECK:STDOUT:   %tuple.loc34: %tuple.type.2 = tuple_value (%.loc34_4, %.loc34_7)
// CHECK:STDOUT:   %.loc34_8.2: %tuple.type.2 = converted %.loc34_8.1, %tuple.loc34
// CHECK:STDOUT:   %.loc39_3: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc39_3: init type = call constants.%Int(%.loc39_3) [template = constants.%i32]
// CHECK:STDOUT:   %.loc39_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc39_9: init type = call constants.%Int(%.loc39_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc39_12.1: type = value_of_initializer %int.make_type_signed.loc39_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc39_12.2: type = converted %int.make_type_signed.loc39_9, %.loc39_12.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc39_12.3: type = ptr_type %i32 [template = constants.%.42]
// CHECK:STDOUT:   assign %int.make_type_signed.loc39_3, %.loc39_12.3
// CHECK:STDOUT:   %.loc44_9: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_17: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc44_18.1: %.43 = struct_literal (%.loc44_9, %.loc44_17)
// CHECK:STDOUT:   %.loc44_28: Core.IntLiteral = int_value 3 [template = constants.%.31]
// CHECK:STDOUT:   %.loc44_36: Core.IntLiteral = int_value 4 [template = constants.%.32]
// CHECK:STDOUT:   %.loc44_37.1: %.43 = struct_literal (%.loc44_28, %.loc44_36)
// CHECK:STDOUT:   %.loc44_37.2: Core.IntLiteral = struct_access %.loc44_18.1, element0
// CHECK:STDOUT:   %.loc44_37.3: init Core.IntLiteral = initialize_from %.loc44_28 to %.loc44_37.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc44_37.4: Core.IntLiteral = struct_access %.loc44_18.1, element1
// CHECK:STDOUT:   %.loc44_37.5: init Core.IntLiteral = initialize_from %.loc44_36 to %.loc44_37.4 [template = constants.%.32]
// CHECK:STDOUT:   %.loc44_37.6: init %.43 = struct_init (%.loc44_37.3, %.loc44_37.5) to %.loc44_18.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc44_20: init %.43 = converted %.loc44_37.1, %.loc44_37.6 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %.loc44_18.1, %.loc44_20
// CHECK:STDOUT:   %struct: %.43 = struct_value (%.loc44_9, %.loc44_17) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc44_18.2: %.43 = converted %.loc44_18.1, %struct [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc49_7: bool = bool_literal true [template = constants.%.45]
// CHECK:STDOUT:   if %.loc49_7 br !if.expr.then.loc49 else br !if.expr.else.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc49:
// CHECK:STDOUT:   %.loc49_17: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc49_12.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc49_12.2: <bound method> = bound_method %.loc49_17, %.loc49_12.1 [template = constants.%.28]
// CHECK:STDOUT:   %.loc49_12.3: <specific function> = specific_function %.loc49_12.2, @Convert.2(constants.%.1) [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc49_12: init %i32 = call %.loc49_12.3(%.loc49_17) [template = constants.%.30]
// CHECK:STDOUT:   %.loc49_12.4: %i32 = value_of_initializer %int.convert_checked.loc49_12 [template = constants.%.30]
// CHECK:STDOUT:   %.loc49_12.5: %i32 = converted %.loc49_17, %.loc49_12.4 [template = constants.%.30]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_12.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc49:
// CHECK:STDOUT:   %.loc49_24: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc49_19.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc49_19.2: <bound method> = bound_method %.loc49_24, %.loc49_19.1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc49_19.3: <specific function> = specific_function %.loc49_19.2, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc49_19: init %i32 = call %.loc49_19.3(%.loc49_24) [template = constants.%.41]
// CHECK:STDOUT:   %.loc49_19.4: %i32 = value_of_initializer %int.convert_checked.loc49_19 [template = constants.%.41]
// CHECK:STDOUT:   %.loc49_19.5: %i32 = converted %.loc49_24, %.loc49_19.4 [template = constants.%.41]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_19.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc49:
// CHECK:STDOUT:   %.loc49_4: %i32 = block_arg !if.expr.result.loc49 [template = constants.%.30]
// CHECK:STDOUT:   %.loc49_29: Core.IntLiteral = int_value 3 [template = constants.%.31]
// CHECK:STDOUT:   %.loc49_27.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc49_27.2: <bound method> = bound_method %.loc49_29, %.loc49_27.1 [template = constants.%.46]
// CHECK:STDOUT:   %.loc49_27.3: <specific function> = specific_function %.loc49_27.2, @Convert.2(constants.%.1) [template = constants.%.47]
// CHECK:STDOUT:   %int.convert_checked.loc49_27: init %i32 = call %.loc49_27.3(%.loc49_29) [template = constants.%.48]
// CHECK:STDOUT:   %.loc49_27.4: init %i32 = converted %.loc49_29, %int.convert_checked.loc49_27 [template = constants.%.48]
// CHECK:STDOUT:   assign %.loc49_4, %.loc49_27.4
// CHECK:STDOUT:   %.loc52_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc52: init type = call constants.%Int(%.loc52_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc52_10.2: type = value_of_initializer %int.make_type_signed.loc52 [template = constants.%i32]
// CHECK:STDOUT:   %.loc52_10.3: type = converted %int.make_type_signed.loc52, %.loc52_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc56_7: bool = bool_literal true [template = constants.%.45]
// CHECK:STDOUT:   if %.loc56_7 br !if.expr.then.loc56 else br !if.expr.else.loc56
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc56:
// CHECK:STDOUT:   %a.ref.loc56_17: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc56_17: %i32 = bind_value %a.ref.loc56_17
// CHECK:STDOUT:   br !if.expr.result.loc56(%.loc56_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc56:
// CHECK:STDOUT:   %a.ref.loc56_24: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc56_24: %i32 = bind_value %a.ref.loc56_24
// CHECK:STDOUT:   br !if.expr.result.loc56(%.loc56_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc56:
// CHECK:STDOUT:   %.loc56_4: %i32 = block_arg !if.expr.result.loc56
// CHECK:STDOUT:   %.loc56_29: Core.IntLiteral = int_value 10 [template = constants.%.49]
// CHECK:STDOUT:   %.loc56_27.1: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc56_27.2: <bound method> = bound_method %.loc56_29, %.loc56_27.1 [template = constants.%.50]
// CHECK:STDOUT:   %.loc56_27.3: <specific function> = specific_function %.loc56_27.2, @Convert.2(constants.%.1) [template = constants.%.51]
// CHECK:STDOUT:   %int.convert_checked.loc56: init %i32 = call %.loc56_27.3(%.loc56_29) [template = constants.%.52]
// CHECK:STDOUT:   %.loc56_27.4: init %i32 = converted %.loc56_29, %int.convert_checked.loc56 [template = constants.%.52]
// CHECK:STDOUT:   assign %.loc56_4, %.loc56_27.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
