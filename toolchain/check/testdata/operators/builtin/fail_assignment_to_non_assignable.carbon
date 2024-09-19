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
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  1 = 2;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  F() = 1;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  i32 = i32*;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: error: expression is not assignable
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:3: error: expression is not assignable
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  (if true then a else a) = 10;
}

// CHECK:STDOUT: --- fail_assignment_to_non_assignable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %tuple.1: %.4 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %tuple.2: %.4 = tuple_value (%.2, %.3) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.10 [template]
// CHECK:STDOUT:   %struct.1: %.10 = struct_value (%.5, %.6) [template]
// CHECK:STDOUT:   %struct.2: %.10 = struct_value (%.2, %.3) [template]
// CHECK:STDOUT:   %.12: bool = bool_literal true [template]
// CHECK:STDOUT:   %.13: i32 = int_literal 10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_3: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_7: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   assign %.loc18_3, %.loc18_7
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %F.call, %.loc23
// CHECK:STDOUT:   %.loc28_4: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc28_7: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc28_8.1: %.4 = tuple_literal (%.loc28_4, %.loc28_7)
// CHECK:STDOUT:   %.loc28_13: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc28_16: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc28_17.1: %.4 = tuple_literal (%.loc28_13, %.loc28_16)
// CHECK:STDOUT:   %.loc28_17.2: i32 = tuple_access %.loc28_8.1, element0
// CHECK:STDOUT:   %.loc28_17.3: init i32 = initialize_from %.loc28_13 to %.loc28_17.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc28_17.4: i32 = tuple_access %.loc28_8.1, element1
// CHECK:STDOUT:   %.loc28_17.5: init i32 = initialize_from %.loc28_16 to %.loc28_17.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc28_17.6: init %.4 = tuple_init (%.loc28_17.3, %.loc28_17.5) to %.loc28_8.1 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc28_10: init %.4 = converted %.loc28_17.1, %.loc28_17.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign %.loc28_8.1, %.loc28_10
// CHECK:STDOUT:   %tuple.loc28: %.4 = tuple_value (%.loc28_4, %.loc28_7) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc28_8.2: %.4 = converted %.loc28_8.1, %tuple.loc28 [template = constants.%tuple.2]
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc29_10.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_10.2: type = converted %int.make_type_32.loc29, %.loc29_10.1 [template = i32]
// CHECK:STDOUT:   %n.var: ref i32 = var n
// CHECK:STDOUT:   %n: ref i32 = bind_name n, %n.var
// CHECK:STDOUT:   %.loc29_16: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   assign %n.var, %.loc29_16
// CHECK:STDOUT:   %n.ref.loc34_4: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc34_7: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc34_8.1: %.4 = tuple_literal (%n.ref.loc34_4, %n.ref.loc34_7)
// CHECK:STDOUT:   %.loc34_13: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_16: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc34_17.1: %.4 = tuple_literal (%.loc34_13, %.loc34_16)
// CHECK:STDOUT:   %.loc34_17.2: i32 = tuple_access %.loc34_8.1, element0
// CHECK:STDOUT:   %.loc34_17.3: init i32 = initialize_from %.loc34_13 to %.loc34_17.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc34_17.4: i32 = tuple_access %.loc34_8.1, element1
// CHECK:STDOUT:   %.loc34_17.5: init i32 = initialize_from %.loc34_16 to %.loc34_17.4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc34_17.6: init %.4 = tuple_init (%.loc34_17.3, %.loc34_17.5) to %.loc34_8.1 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc34_10: init %.4 = converted %.loc34_17.1, %.loc34_17.6 [template = constants.%tuple.2]
// CHECK:STDOUT:   assign %.loc34_8.1, %.loc34_10
// CHECK:STDOUT:   %.loc34_4: i32 = bind_value %n.ref.loc34_4
// CHECK:STDOUT:   %.loc34_7: i32 = bind_value %n.ref.loc34_7
// CHECK:STDOUT:   %tuple.loc34: %.4 = tuple_value (%.loc34_4, %.loc34_7)
// CHECK:STDOUT:   %.loc34_8.2: %.4 = converted %.loc34_8.1, %tuple.loc34
// CHECK:STDOUT:   %int.make_type_32.loc39_3: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc39_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc39_12.1: type = value_of_initializer %int.make_type_32.loc39_9 [template = i32]
// CHECK:STDOUT:   %.loc39_12.2: type = converted %int.make_type_32.loc39_9, %.loc39_12.1 [template = i32]
// CHECK:STDOUT:   %.loc39_12.3: type = ptr_type i32 [template = constants.%.9]
// CHECK:STDOUT:   assign %int.make_type_32.loc39_3, %.loc39_12.3
// CHECK:STDOUT:   %.loc44_9: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc44_17: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc44_18.1: %.10 = struct_literal (%.loc44_9, %.loc44_17)
// CHECK:STDOUT:   %.loc44_28: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc44_36: i32 = int_literal 4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc44_37.1: %.10 = struct_literal (%.loc44_28, %.loc44_36)
// CHECK:STDOUT:   %.loc44_37.2: i32 = struct_access %.loc44_18.1, element0
// CHECK:STDOUT:   %.loc44_37.3: init i32 = initialize_from %.loc44_28 to %.loc44_37.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc44_37.4: i32 = struct_access %.loc44_18.1, element1
// CHECK:STDOUT:   %.loc44_37.5: init i32 = initialize_from %.loc44_36 to %.loc44_37.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc44_37.6: init %.10 = struct_init (%.loc44_37.3, %.loc44_37.5) to %.loc44_18.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc44_20: init %.10 = converted %.loc44_37.1, %.loc44_37.6 [template = constants.%struct.1]
// CHECK:STDOUT:   assign %.loc44_18.1, %.loc44_20
// CHECK:STDOUT:   %struct: %.10 = struct_value (%.loc44_9, %.loc44_17) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc44_18.2: %.10 = converted %.loc44_18.1, %struct [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc49_7: bool = bool_literal true [template = constants.%.12]
// CHECK:STDOUT:   if %.loc49_7 br !if.expr.then.loc49 else br !if.expr.else.loc49
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc49:
// CHECK:STDOUT:   %.loc49_17: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc49:
// CHECK:STDOUT:   %.loc49_24: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   br !if.expr.result.loc49(%.loc49_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc49:
// CHECK:STDOUT:   %.loc49_4: i32 = block_arg !if.expr.result.loc49 [template = constants.%.2]
// CHECK:STDOUT:   %.loc49_29: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   assign %.loc49_4, %.loc49_29
// CHECK:STDOUT:   %int.make_type_32.loc52: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc52_10.1: type = value_of_initializer %int.make_type_32.loc52 [template = i32]
// CHECK:STDOUT:   %.loc52_10.2: type = converted %int.make_type_32.loc52, %.loc52_10.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc56_7: bool = bool_literal true [template = constants.%.12]
// CHECK:STDOUT:   if %.loc56_7 br !if.expr.then.loc56 else br !if.expr.else.loc56
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc56:
// CHECK:STDOUT:   %a.ref.loc56_17: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc56_17: i32 = bind_value %a.ref.loc56_17
// CHECK:STDOUT:   br !if.expr.result.loc56(%.loc56_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc56:
// CHECK:STDOUT:   %a.ref.loc56_24: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc56_24: i32 = bind_value %a.ref.loc56_24
// CHECK:STDOUT:   br !if.expr.result.loc56(%.loc56_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc56:
// CHECK:STDOUT:   %.loc56_4: i32 = block_arg !if.expr.result.loc56
// CHECK:STDOUT:   %.loc56_29: i32 = int_literal 10 [template = constants.%.13]
// CHECK:STDOUT:   assign %.loc56_4, %.loc56_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
