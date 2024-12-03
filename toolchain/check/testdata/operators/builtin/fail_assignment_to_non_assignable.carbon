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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.1 = tuple_value (%int_3.1, %int_4) [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.1 = tuple_value (%int_1.1, %int_2.1) [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.3: %tuple.type.2 = tuple_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.1: %struct_type.x.y = struct_value (%int_3.1, %int_4) [template]
// CHECK:STDOUT:   %struct.2: %struct_type.x.y = struct_value (%int_1.1, %int_2.1) [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_10.1: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_10.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_10.2: %i32 = int_value 10 [template]
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_signed, %.loc11_11.1 [template = constants.%i32]
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
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc18: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   assign %int_1.loc18, %int_2.loc18
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc23: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc23: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc23: <specific function> = specific_function %Convert.bound.loc23, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %Convert.specific_fn.loc23(%int_1.loc23) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc23: init %i32 = converted %int_1.loc23, %int.convert_checked.loc23 [template = constants.%int_1.2]
// CHECK:STDOUT:   assign %F.call, %.loc23
// CHECK:STDOUT:   %int_1.loc28: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc28: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc28_8.1: %tuple.type.1 = tuple_literal (%int_1.loc28, %int_2.loc28)
// CHECK:STDOUT:   %int_3.loc28: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_4.loc28: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc28_17.1: %tuple.type.1 = tuple_literal (%int_3.loc28, %int_4.loc28)
// CHECK:STDOUT:   %tuple.elem0.loc28: Core.IntLiteral = tuple_access %.loc28_8.1, element0
// CHECK:STDOUT:   %.loc28_17.2: init Core.IntLiteral = initialize_from %int_3.loc28 to %tuple.elem0.loc28 [template = constants.%int_3.1]
// CHECK:STDOUT:   %tuple.elem1.loc28: Core.IntLiteral = tuple_access %.loc28_8.1, element1
// CHECK:STDOUT:   %.loc28_17.3: init Core.IntLiteral = initialize_from %int_4.loc28 to %tuple.elem1.loc28 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc28_17.4: init %tuple.type.1 = tuple_init (%.loc28_17.2, %.loc28_17.3) to %.loc28_8.1 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc28_10: init %tuple.type.1 = converted %.loc28_17.1, %.loc28_17.4 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign %.loc28_8.1, %.loc28_10
// CHECK:STDOUT:   %tuple.loc28: %tuple.type.1 = tuple_value (%int_1.loc28, %int_2.loc28) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc28_8.2: %tuple.type.1 = converted %.loc28_8.1, %tuple.loc28 [template = constants.%tuple.2]
// CHECK:STDOUT:   %int_32.loc29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc29: init type = call constants.%Int(%int_32.loc29) [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_10.1: type = value_of_initializer %int.make_type_signed.loc29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc29_10.2: type = converted %int.make_type_signed.loc29, %.loc29_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %n.var: ref %i32 = var n
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %impl.elem0.loc29: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc29: <bound method> = bound_method %int_0, %impl.elem0.loc29 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc29: <specific function> = specific_function %Convert.bound.loc29, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %i32 = call %Convert.specific_fn.loc29(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc29_17: init %i32 = converted %int_0, %int.convert_checked.loc29 [template = constants.%int_0.2]
// CHECK:STDOUT:   assign %n.var, %.loc29_17
// CHECK:STDOUT:   %n.ref.loc34_4: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc34_7: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc34_8.1: %tuple.type.2 = tuple_literal (%n.ref.loc34_4, %n.ref.loc34_7)
// CHECK:STDOUT:   %int_1.loc34: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc34: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc34_17.1: %tuple.type.1 = tuple_literal (%int_1.loc34, %int_2.loc34)
// CHECK:STDOUT:   %impl.elem0.loc34_17.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc34_17.1: <bound method> = bound_method %int_1.loc34, %impl.elem0.loc34_17.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_17.1: <specific function> = specific_function %Convert.bound.loc34_17.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc34_17.1: init %i32 = call %Convert.specific_fn.loc34_17.1(%int_1.loc34) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc34_17.2: init %i32 = converted %int_1.loc34, %int.convert_checked.loc34_17.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple.elem0.loc34: %i32 = tuple_access %.loc34_8.1, element0
// CHECK:STDOUT:   %.loc34_17.3: init %i32 = initialize_from %.loc34_17.2 to %tuple.elem0.loc34 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc34_17.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc34_17.2: <bound method> = bound_method %int_2.loc34, %impl.elem0.loc34_17.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc34_17.2: <specific function> = specific_function %Convert.bound.loc34_17.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc34_17.2: init %i32 = call %Convert.specific_fn.loc34_17.2(%int_2.loc34) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc34_17.4: init %i32 = converted %int_2.loc34, %int.convert_checked.loc34_17.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.elem1.loc34: %i32 = tuple_access %.loc34_8.1, element1
// CHECK:STDOUT:   %.loc34_17.5: init %i32 = initialize_from %.loc34_17.4 to %tuple.elem1.loc34 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc34_17.6: init %tuple.type.2 = tuple_init (%.loc34_17.3, %.loc34_17.5) to %.loc34_8.1 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc34_10: init %tuple.type.2 = converted %.loc34_17.1, %.loc34_17.6 [template = constants.%tuple.3]
// CHECK:STDOUT:   assign %.loc34_8.1, %.loc34_10
// CHECK:STDOUT:   %.loc34_4: %i32 = bind_value %n.ref.loc34_4
// CHECK:STDOUT:   %.loc34_7: %i32 = bind_value %n.ref.loc34_7
// CHECK:STDOUT:   %tuple.loc34: %tuple.type.2 = tuple_value (%.loc34_4, %.loc34_7)
// CHECK:STDOUT:   %.loc34_8.2: %tuple.type.2 = converted %.loc34_8.1, %tuple.loc34
// CHECK:STDOUT:   %int_32.loc39_3: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc39_3: init type = call constants.%Int(%int_32.loc39_3) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc39_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc39_9: init type = call constants.%Int(%int_32.loc39_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc39_12.1: type = value_of_initializer %int.make_type_signed.loc39_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc39_12.2: type = converted %int.make_type_signed.loc39_9, %.loc39_12.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [template = constants.%ptr.3]
// CHECK:STDOUT:   assign %int.make_type_signed.loc39_3, %ptr
// CHECK:STDOUT:   %int_1.loc44: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc44: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc44_18.1: %struct_type.x.y = struct_literal (%int_1.loc44, %int_2.loc44)
// CHECK:STDOUT:   %int_3.loc44: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_4.loc44: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc44_37.1: %struct_type.x.y = struct_literal (%int_3.loc44, %int_4.loc44)
// CHECK:STDOUT:   %.loc44_37.2: Core.IntLiteral = struct_access %.loc44_18.1, element0
// CHECK:STDOUT:   %.loc44_37.3: init Core.IntLiteral = initialize_from %int_3.loc44 to %.loc44_37.2 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc44_37.4: Core.IntLiteral = struct_access %.loc44_18.1, element1
// CHECK:STDOUT:   %.loc44_37.5: init Core.IntLiteral = initialize_from %int_4.loc44 to %.loc44_37.4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc44_37.6: init %struct_type.x.y = struct_init (%.loc44_37.3, %.loc44_37.5) to %.loc44_18.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc44_20: init %struct_type.x.y = converted %.loc44_37.1, %.loc44_37.6 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %.loc44_18.1, %.loc44_20
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%int_1.loc44, %int_2.loc44) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc44_18.2: %struct_type.x.y = converted %.loc44_18.1, %struct [template = constants.%struct.2]
// CHECK:STDOUT:   %true.loc49: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true.loc49 br !if.expr.then.loc49 else br !if.expr.else.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc49:
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32.loc49: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc49: init type = call constants.%Int(%int_32.loc49) [template = constants.%i32]
// CHECK:STDOUT:   %.loc49_12.1: type = value_of_initializer %int.make_type_signed.loc49 [template = constants.%i32]
// CHECK:STDOUT:   %.loc49_12.2: type = converted %int.make_type_signed.loc49, %.loc49_12.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc49_12: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc49_12: <bound method> = bound_method %int_1.loc49, %impl.elem0.loc49_12 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc49_12: <specific function> = specific_function %Convert.bound.loc49_12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc49_12: init %i32 = call %Convert.specific_fn.loc49_12(%int_1.loc49) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc49_12.3: %i32 = value_of_initializer %int.convert_checked.loc49_12 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc49_12.4: %i32 = converted %int_1.loc49, %.loc49_12.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_12.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc49:
// CHECK:STDOUT:   %int_2.loc49: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %impl.elem0.loc49_19: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc49_19: <bound method> = bound_method %int_2.loc49, %impl.elem0.loc49_19 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc49_19: <specific function> = specific_function %Convert.bound.loc49_19, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc49_19: init %i32 = call %Convert.specific_fn.loc49_19(%int_2.loc49) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc49_19.1: %i32 = value_of_initializer %int.convert_checked.loc49_19 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc49_19.2: %i32 = converted %int_2.loc49, %.loc49_19.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc49:
// CHECK:STDOUT:   %.loc49_4: %i32 = block_arg !if.expr.result.loc49 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int_3.loc49: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %impl.elem0.loc49_27: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc49_27: <bound method> = bound_method %int_3.loc49, %impl.elem0.loc49_27 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc49_27: <specific function> = specific_function %Convert.bound.loc49_27, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc49_27: init %i32 = call %Convert.specific_fn.loc49_27(%int_3.loc49) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc49_27: init %i32 = converted %int_3.loc49, %int.convert_checked.loc49_27 [template = constants.%int_3.2]
// CHECK:STDOUT:   assign %.loc49_4, %.loc49_27
// CHECK:STDOUT:   %int_32.loc52: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc52: init type = call constants.%Int(%int_32.loc52) [template = constants.%i32]
// CHECK:STDOUT:   %.loc52_10.1: type = value_of_initializer %int.make_type_signed.loc52 [template = constants.%i32]
// CHECK:STDOUT:   %.loc52_10.2: type = converted %int.make_type_signed.loc52, %.loc52_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %true.loc56: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true.loc56 br !if.expr.then.loc56 else br !if.expr.else.loc56
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
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10.1]
// CHECK:STDOUT:   %impl.elem0.loc56: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc56: <bound method> = bound_method %int_10, %impl.elem0.loc56 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc56: <specific function> = specific_function %Convert.bound.loc56, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc56: init %i32 = call %Convert.specific_fn.loc56(%int_10) [template = constants.%int_10.2]
// CHECK:STDOUT:   %.loc56_27: init %i32 = converted %int_10, %int.convert_checked.loc56 [template = constants.%int_10.2]
// CHECK:STDOUT:   assign %.loc56_4, %.loc56_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
