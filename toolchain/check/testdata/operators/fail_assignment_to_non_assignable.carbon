// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32;

fn Main() {
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  1 = 2;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:5: ERROR: Expression is not assignable.
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:     ^
  F() = 1;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:8: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:        ^
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:8: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:        ^
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^
  i32 = i32*;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:18: ERROR: Expression is not assignable.
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:                  ^
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:25: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:                         ^
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:25: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:                         ^
  (if true then a else a) = 10;
}

// CHECK:STDOUT: file "fail_assignment_to_non_assignable.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_3: i32 = int_literal 1
// CHECK:STDOUT:   %.loc13_7: i32 = int_literal 2
// CHECK:STDOUT:   assign %.loc13_3, %.loc13_7
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc17_4: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc17_9: i32 = int_literal 1
// CHECK:STDOUT:   assign %.loc17_4, %.loc17_9
// CHECK:STDOUT:   %.loc21_4: i32 = int_literal 1
// CHECK:STDOUT:   %.loc21_7: i32 = int_literal 2
// CHECK:STDOUT:   %.loc21_8.1: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc21_8.2: (i32, i32) = tuple_literal (%.loc21_4, %.loc21_7)
// CHECK:STDOUT:   %.loc21_13: i32 = int_literal 3
// CHECK:STDOUT:   %.loc21_16: i32 = int_literal 4
// CHECK:STDOUT:   %.loc21_17.1: (i32, i32) = tuple_literal (%.loc21_13, %.loc21_16)
// CHECK:STDOUT:   %.loc21_8.3: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %.loc21_17.2: i32 = tuple_access %.loc21_8.2, member0
// CHECK:STDOUT:   %.loc21_17.3: init i32 = initialize_from %.loc21_13 to %.loc21_17.2
// CHECK:STDOUT:   %.loc21_17.4: i32 = tuple_access %.loc21_8.2, member1
// CHECK:STDOUT:   %.loc21_17.5: init i32 = initialize_from %.loc21_16 to %.loc21_17.4
// CHECK:STDOUT:   %.loc21_17.6: init (i32, i32) = tuple_init %.loc21_17.1, (%.loc21_17.3, %.loc21_17.5)
// CHECK:STDOUT:   assign %.loc21_8.2, %.loc21_17.6
// CHECK:STDOUT:   %.loc21_8.4: (i32, i32) = tuple_value %.loc21_8.2, (%.loc21_4, %.loc21_7)
// CHECK:STDOUT:   %n: ref i32 = var "n"
// CHECK:STDOUT:   %.loc22: i32 = int_literal 0
// CHECK:STDOUT:   assign %n, %.loc22
// CHECK:STDOUT:   %n.ref.loc26_4: ref i32 = name_reference "n", %n
// CHECK:STDOUT:   %n.ref.loc26_7: ref i32 = name_reference "n", %n
// CHECK:STDOUT:   %.loc26_8.1: (i32, i32) = tuple_literal (%n.ref.loc26_4, %n.ref.loc26_7)
// CHECK:STDOUT:   %.loc26_13: i32 = int_literal 1
// CHECK:STDOUT:   %.loc26_16: i32 = int_literal 2
// CHECK:STDOUT:   %.loc26_17.1: (i32, i32) = tuple_literal (%.loc26_13, %.loc26_16)
// CHECK:STDOUT:   %.loc26_17.2: i32 = tuple_access %.loc26_8.1, member0
// CHECK:STDOUT:   %.loc26_17.3: init i32 = initialize_from %.loc26_13 to %.loc26_17.2
// CHECK:STDOUT:   %.loc26_17.4: i32 = tuple_access %.loc26_8.1, member1
// CHECK:STDOUT:   %.loc26_17.5: init i32 = initialize_from %.loc26_16 to %.loc26_17.4
// CHECK:STDOUT:   %.loc26_17.6: init (i32, i32) = tuple_init %.loc26_17.1, (%.loc26_17.3, %.loc26_17.5)
// CHECK:STDOUT:   assign %.loc26_8.1, %.loc26_17.6
// CHECK:STDOUT:   %.loc26_4: i32 = bind_value %n.ref.loc26_4
// CHECK:STDOUT:   %.loc26_7: i32 = bind_value %n.ref.loc26_7
// CHECK:STDOUT:   %.loc26_8.2: (i32, i32) = tuple_value %.loc26_8.1, (%.loc26_4, %.loc26_7)
// CHECK:STDOUT:   %.loc30: type = ptr_type i32
// CHECK:STDOUT:   assign i32, %.loc30
// CHECK:STDOUT:   %.loc34_9: i32 = int_literal 1
// CHECK:STDOUT:   %.loc34_17: i32 = int_literal 2
// CHECK:STDOUT:   %.loc34_18.1: type = struct_type {.x: i32, .y: i32}
// CHECK:STDOUT:   %.loc34_18.2: {.x: i32, .y: i32} = struct_literal (%.loc34_9, %.loc34_17)
// CHECK:STDOUT:   %.loc34_28: i32 = int_literal 3
// CHECK:STDOUT:   %.loc34_36: i32 = int_literal 4
// CHECK:STDOUT:   %.loc34_37.1: {.x: i32, .y: i32} = struct_literal (%.loc34_28, %.loc34_36)
// CHECK:STDOUT:   %.loc34_18.3: type = ptr_type {.x: i32, .y: i32}
// CHECK:STDOUT:   %.loc34_37.2: i32 = struct_access %.loc34_18.2, member0
// CHECK:STDOUT:   %.loc34_37.3: init i32 = initialize_from %.loc34_28 to %.loc34_37.2
// CHECK:STDOUT:   %.loc34_37.4: i32 = struct_access %.loc34_18.2, member1
// CHECK:STDOUT:   %.loc34_37.5: init i32 = initialize_from %.loc34_36 to %.loc34_37.4
// CHECK:STDOUT:   %.loc34_37.6: init {.x: i32, .y: i32} = struct_init %.loc34_37.1, (%.loc34_37.3, %.loc34_37.5)
// CHECK:STDOUT:   assign %.loc34_18.2, %.loc34_37.6
// CHECK:STDOUT:   %.loc34_18.4: {.x: i32, .y: i32} = struct_value %.loc34_18.2, (%.loc34_9, %.loc34_17)
// CHECK:STDOUT:   %.loc38_7: bool = bool_literal true
// CHECK:STDOUT:   if %.loc38_7 br !if.expr.then.loc38 else br !if.expr.else.loc38
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc38:
// CHECK:STDOUT:   %.loc38_17: i32 = int_literal 1
// CHECK:STDOUT:   br !if.expr.result.loc38(%.loc38_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc38:
// CHECK:STDOUT:   %.loc38_24: i32 = int_literal 2
// CHECK:STDOUT:   br !if.expr.result.loc38(%.loc38_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc38:
// CHECK:STDOUT:   %.loc38_4: i32 = block_arg !if.expr.result.loc38
// CHECK:STDOUT:   %.loc38_29: i32 = int_literal 3
// CHECK:STDOUT:   assign %.loc38_4, %.loc38_29
// CHECK:STDOUT:   %a: ref i32 = var "a"
// CHECK:STDOUT:   %.loc45_7: bool = bool_literal true
// CHECK:STDOUT:   if %.loc45_7 br !if.expr.then.loc45 else br !if.expr.else.loc45
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc45:
// CHECK:STDOUT:   %a.ref.loc45_17: ref i32 = name_reference "a", %a
// CHECK:STDOUT:   %.loc45_17: i32 = bind_value %a.ref.loc45_17
// CHECK:STDOUT:   br !if.expr.result.loc45(%.loc45_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc45:
// CHECK:STDOUT:   %a.ref.loc45_24: ref i32 = name_reference "a", %a
// CHECK:STDOUT:   %.loc45_24: i32 = bind_value %a.ref.loc45_24
// CHECK:STDOUT:   br !if.expr.result.loc45(%.loc45_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc45:
// CHECK:STDOUT:   %.loc45_4: i32 = block_arg !if.expr.result.loc45
// CHECK:STDOUT:   %.loc45_29: i32 = int_literal 10
// CHECK:STDOUT:   assign %.loc45_4, %.loc45_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
