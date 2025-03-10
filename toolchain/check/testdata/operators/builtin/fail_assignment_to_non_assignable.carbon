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
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (if true then a else a) = 10;
}

// CHECK:STDOUT: --- fail_assignment_to_non_assignable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %tuple.302: %tuple.type.f94 = tuple_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Convert.bound.d04: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b6e: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.21c: %tuple.type.d07 = tuple_value (%int_1.5d2, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.a0d: %struct_type.x.y = struct_value (%int_3.1ba, %int_4) [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Convert.bound.491: <bound method> = bound_method %int_10.64f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.5ba: <bound method> = bound_method %int_10.64f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.265: %i32 = int_value 10 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc18: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   assign %int_1.loc18, %int_2.loc18
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   %int_1.loc23: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc23: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc23_7.1: <bound method> = bound_method %int_1.loc23, %impl.elem0.loc23 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_7.2: <bound method> = bound_method %int_1.loc23, %specific_fn.loc23 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %bound_method.loc23_7.2(%int_1.loc23) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc23: init %i32 = converted %int_1.loc23, %int.convert_checked.loc23 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %F.call, %.loc23
// CHECK:STDOUT:   %int_1.loc28: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc28: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc28_8.1: %tuple.type.f94 = tuple_literal (%int_1.loc28, %int_2.loc28)
// CHECK:STDOUT:   %int_3.loc28: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc28: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc28_17.1: %tuple.type.f94 = tuple_literal (%int_3.loc28, %int_4.loc28)
// CHECK:STDOUT:   %tuple.elem0.loc28: Core.IntLiteral = tuple_access %.loc28_8.1, element0
// CHECK:STDOUT:   %.loc28_17.2: init Core.IntLiteral = initialize_from %int_3.loc28 to %tuple.elem0.loc28 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %tuple.elem1.loc28: Core.IntLiteral = tuple_access %.loc28_8.1, element1
// CHECK:STDOUT:   %.loc28_17.3: init Core.IntLiteral = initialize_from %int_4.loc28 to %tuple.elem1.loc28 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc28_17.4: init %tuple.type.f94 = tuple_init (%.loc28_17.2, %.loc28_17.3) to %.loc28_8.1 [concrete = constants.%tuple.302]
// CHECK:STDOUT:   %.loc28_10: init %tuple.type.f94 = converted %.loc28_17.1, %.loc28_17.4 [concrete = constants.%tuple.302]
// CHECK:STDOUT:   assign %.loc28_8.1, %.loc28_10
// CHECK:STDOUT:   %tuple.loc28: %tuple.type.f94 = tuple_value (%int_1.loc28, %int_2.loc28) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   %.loc28_8.2: %tuple.type.f94 = converted %.loc28_8.1, %tuple.loc28 [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %.loc29_3.1: %i32 = var_pattern %n.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc29: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc29_3.1: <bound method> = bound_method %int_0, %impl.elem0.loc29 [concrete = constants.%Convert.bound.d04]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_3.2: <bound method> = bound_method %int_0, %specific_fn.loc29 [concrete = constants.%bound_method.b6e]
// CHECK:STDOUT:   %int.convert_checked.loc29: init %i32 = call %bound_method.loc29_3.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc29_3.2: init %i32 = converted %int_0, %int.convert_checked.loc29 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %n.var, %.loc29_3.2
// CHECK:STDOUT:   %.loc29_10: type = splice_block %i32.loc29 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   %n.ref.loc34_4: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc34_7: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc34_8.1: %tuple.type.d07 = tuple_literal (%n.ref.loc34_4, %n.ref.loc34_7)
// CHECK:STDOUT:   %int_1.loc34: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc34: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc34_17.1: %tuple.type.f94 = tuple_literal (%int_1.loc34, %int_2.loc34)
// CHECK:STDOUT:   %impl.elem0.loc34_17.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc34_17.1: <bound method> = bound_method %int_1.loc34, %impl.elem0.loc34_17.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc34_17.1: <specific function> = specific_function %impl.elem0.loc34_17.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_17.2: <bound method> = bound_method %int_1.loc34, %specific_fn.loc34_17.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc34_17.1: init %i32 = call %bound_method.loc34_17.2(%int_1.loc34) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc34_17.2: init %i32 = converted %int_1.loc34, %int.convert_checked.loc34_17.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0.loc34: %i32 = tuple_access %.loc34_8.1, element0
// CHECK:STDOUT:   %.loc34_17.3: init %i32 = initialize_from %.loc34_17.2 to %tuple.elem0.loc34 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc34_17.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc34_17.3: <bound method> = bound_method %int_2.loc34, %impl.elem0.loc34_17.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc34_17.2: <specific function> = specific_function %impl.elem0.loc34_17.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc34_17.4: <bound method> = bound_method %int_2.loc34, %specific_fn.loc34_17.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc34_17.2: init %i32 = call %bound_method.loc34_17.4(%int_2.loc34) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc34_17.4: init %i32 = converted %int_2.loc34, %int.convert_checked.loc34_17.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc34: %i32 = tuple_access %.loc34_8.1, element1
// CHECK:STDOUT:   %.loc34_17.5: init %i32 = initialize_from %.loc34_17.4 to %tuple.elem1.loc34 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc34_17.6: init %tuple.type.d07 = tuple_init (%.loc34_17.3, %.loc34_17.5) to %.loc34_8.1 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   %.loc34_10: init %tuple.type.d07 = converted %.loc34_17.1, %.loc34_17.6 [concrete = constants.%tuple.21c]
// CHECK:STDOUT:   assign %.loc34_8.1, %.loc34_10
// CHECK:STDOUT:   %.loc34_4: %i32 = bind_value %n.ref.loc34_4
// CHECK:STDOUT:   %.loc34_7: %i32 = bind_value %n.ref.loc34_7
// CHECK:STDOUT:   %tuple.loc34: %tuple.type.d07 = tuple_value (%.loc34_4, %.loc34_7)
// CHECK:STDOUT:   %.loc34_8.2: %tuple.type.d07 = converted %.loc34_8.1, %tuple.loc34
// CHECK:STDOUT:   %int_32.loc39_3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc39_3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_32.loc39_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc39_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [concrete = constants.%ptr.235]
// CHECK:STDOUT:   assign %i32.loc39_3, %ptr
// CHECK:STDOUT:   %int_1.loc44: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc44: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc44_18.1: %struct_type.x.y = struct_literal (%int_1.loc44, %int_2.loc44)
// CHECK:STDOUT:   %int_3.loc44: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc44: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc44_37.1: %struct_type.x.y = struct_literal (%int_3.loc44, %int_4.loc44)
// CHECK:STDOUT:   %.loc44_37.2: Core.IntLiteral = struct_access %.loc44_18.1, element0
// CHECK:STDOUT:   %.loc44_37.3: init Core.IntLiteral = initialize_from %int_3.loc44 to %.loc44_37.2 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc44_37.4: Core.IntLiteral = struct_access %.loc44_18.1, element1
// CHECK:STDOUT:   %.loc44_37.5: init Core.IntLiteral = initialize_from %int_4.loc44 to %.loc44_37.4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc44_37.6: init %struct_type.x.y = struct_init (%.loc44_37.3, %.loc44_37.5) to %.loc44_18.1 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   %.loc44_20: init %struct_type.x.y = converted %.loc44_37.1, %.loc44_37.6 [concrete = constants.%struct.a0d]
// CHECK:STDOUT:   assign %.loc44_18.1, %.loc44_20
// CHECK:STDOUT:   %struct: %struct_type.x.y = struct_value (%int_1.loc44, %int_2.loc44) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %.loc44_18.2: %struct_type.x.y = converted %.loc44_18.1, %struct [concrete = constants.%struct.004]
// CHECK:STDOUT:   %true.loc49: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc49 br !if.expr.then.loc49 else br !if.expr.else.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc49:
// CHECK:STDOUT:   %int_1.loc49: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc49: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc49: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc49_12: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc49_12.1: <bound method> = bound_method %int_1.loc49, %impl.elem0.loc49_12 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc49_12: <specific function> = specific_function %impl.elem0.loc49_12, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_12.2: <bound method> = bound_method %int_1.loc49, %specific_fn.loc49_12 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc49_12: init %i32 = call %bound_method.loc49_12.2(%int_1.loc49) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc49_12.1: %i32 = value_of_initializer %int.convert_checked.loc49_12 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc49_12.2: %i32 = converted %int_1.loc49, %.loc49_12.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_12.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc49:
// CHECK:STDOUT:   %int_2.loc49: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc49_19: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc49_19.1: <bound method> = bound_method %int_2.loc49, %impl.elem0.loc49_19 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc49_19: <specific function> = specific_function %impl.elem0.loc49_19, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_19.2: <bound method> = bound_method %int_2.loc49, %specific_fn.loc49_19 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc49_19: init %i32 = call %bound_method.loc49_19.2(%int_2.loc49) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc49_19.1: %i32 = value_of_initializer %int.convert_checked.loc49_19 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc49_19.2: %i32 = converted %int_2.loc49, %.loc49_19.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc49:
// CHECK:STDOUT:   %.loc49_4: %i32 = block_arg !if.expr.result.loc49 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_3.loc49: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc49_27: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc49_27.1: <bound method> = bound_method %int_3.loc49, %impl.elem0.loc49_27 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc49_27: <specific function> = specific_function %impl.elem0.loc49_27, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc49_27.2: <bound method> = bound_method %int_3.loc49, %specific_fn.loc49_27 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc49_27: init %i32 = call %bound_method.loc49_27.2(%int_3.loc49) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc49_27: init %i32 = converted %int_3.loc49, %int.convert_checked.loc49_27 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %.loc49_4, %.loc49_27
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:     %.loc52_3: %i32 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %.loc52_10: type = splice_block %i32.loc52 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc52: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc52: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %true.loc57: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc57 br !if.expr.then.loc57 else br !if.expr.else.loc57
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc57:
// CHECK:STDOUT:   %a.ref.loc57_17: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc57_17: %i32 = bind_value %a.ref.loc57_17
// CHECK:STDOUT:   br !if.expr.result.loc57(%.loc57_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc57:
// CHECK:STDOUT:   %a.ref.loc57_24: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc57_24: %i32 = bind_value %a.ref.loc57_24
// CHECK:STDOUT:   br !if.expr.result.loc57(%.loc57_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc57:
// CHECK:STDOUT:   %.loc57_4: %i32 = block_arg !if.expr.result.loc57
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %impl.elem0.loc57: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc57_27.1: <bound method> = bound_method %int_10, %impl.elem0.loc57 [concrete = constants.%Convert.bound.491]
// CHECK:STDOUT:   %specific_fn.loc57: <specific function> = specific_function %impl.elem0.loc57, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc57_27.2: <bound method> = bound_method %int_10, %specific_fn.loc57 [concrete = constants.%bound_method.5ba]
// CHECK:STDOUT:   %int.convert_checked.loc57: init %i32 = call %bound_method.loc57_27.2(%int_10) [concrete = constants.%int_10.265]
// CHECK:STDOUT:   %.loc57_27: init %i32 = converted %int_10, %int.convert_checked.loc57 [concrete = constants.%int_10.265]
// CHECK:STDOUT:   assign %.loc57_4, %.loc57_27
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
