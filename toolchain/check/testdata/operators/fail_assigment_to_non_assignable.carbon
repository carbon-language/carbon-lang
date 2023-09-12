// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_assigment_to_non_assignable.carbon" {

fn F() -> i32;
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:3: Expression is not assignable.
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  1 = 2;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_3: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_7: i32 = int_literal 2
  // CHECK:STDOUT:   assign %.loc[[@LINE-3]]_3, %.loc[[@LINE-3]]_7
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:5: Expression is not assignable.
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:     ^
  F() = 1;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_4: init i32 = call @F()
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_9: i32 = int_literal 1
  // CHECK:STDOUT:   assign %.loc[[@LINE-3]]_4, %.loc[[@LINE-3]]_9
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:8: Expression is not assignable.
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:        ^
  (1, 2) = (3, 4);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_4.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_4.2: i32 = stub_reference %.loc[[@LINE-2]]_4.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_7.1: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_7.2: i32 = stub_reference %.loc[[@LINE-4]]_7.1
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_8.1: type = tuple_type (i32, i32)
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_8.2: (i32, i32) = tuple_literal (%.loc[[@LINE-6]]_4.2, %.loc[[@LINE-6]]_7.2)
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_13.1: i32 = int_literal 3
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_13.2: i32 = stub_reference %.loc[[@LINE-8]]_13.1
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_16.1: i32 = int_literal 4
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_16.2: i32 = stub_reference %.loc[[@LINE-10]]_16.1
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_17: (i32, i32) = tuple_literal (%.loc[[@LINE-11]]_13.2, %.loc[[@LINE-11]]_16.2)
  // CHECK:STDOUT:   assign %.loc[[@LINE-12]]_8.2, %.loc[[@LINE-12]]_17
  // CHECK:STDOUT:   %n: ref i32 = var "n"
  var n: i32 = 0;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_16: i32 = int_literal 0
  // CHECK:STDOUT:   assign %n, %.loc[[@LINE-2]]_16
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_7.1: i32 = bind_value %n
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:8: Expression is not assignable.
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:        ^
  (n, n) = (1, 2);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_4: i32 = stub_reference %.loc[[@LINE-8]]_7.1
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_7.2: i32 = bind_value %n
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_7: i32 = stub_reference %.loc[[@LINE-10]]_7.2
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_8: (i32, i32) = tuple_literal (%.loc[[@LINE-4]]_4, %.loc[[@LINE-4]]_7)
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_13.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_13.2: i32 = stub_reference %.loc[[@LINE-6]]_13.1
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_16.1: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_16.2: i32 = stub_reference %.loc[[@LINE-8]]_16.1
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_17: (i32, i32) = tuple_literal (%.loc[[@LINE-9]]_13.2, %.loc[[@LINE-9]]_16.2)
  // CHECK:STDOUT:   assign %.loc[[@LINE-10]]_8, %.loc[[@LINE-10]]_17
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:3: Expression is not assignable.
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^
  i32 = i32*;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]: type = ptr_type i32
  // CHECK:STDOUT:   assign i32, %.loc[[@LINE-2]]
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:18: Expression is not assignable.
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:                  ^
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_9: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_7: i32 = stub_reference %.loc[[@LINE-2]]_9
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_17: i32 = int_literal 2
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_15: i32 = stub_reference %.loc[[@LINE-4]]_17
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_18.1: type = struct_type {.x: i32, .y: i32}
  // CHECK:STDOUT:   %.loc[[@LINE-6]]_18.2: {.x: i32, .y: i32} = struct_literal (%.loc[[@LINE-6]]_7, %.loc[[@LINE-6]]_15)
  // CHECK:STDOUT:   %.loc[[@LINE-7]]_28: i32 = int_literal 3
  // CHECK:STDOUT:   %.loc[[@LINE-8]]_26: i32 = stub_reference %.loc[[@LINE-8]]_28
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_36: i32 = int_literal 4
  // CHECK:STDOUT:   %.loc[[@LINE-10]]_34: i32 = stub_reference %.loc[[@LINE-10]]_36
  // CHECK:STDOUT:   %.loc[[@LINE-11]]_37: {.x: i32, .y: i32} = struct_literal (%.loc[[@LINE-11]]_26, %.loc[[@LINE-11]]_34)
  // CHECK:STDOUT:   assign %.loc[[@LINE-12]]_18.2, %.loc[[@LINE-12]]_37
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:25: Expression is not assignable.
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:                         ^
  (if true then 1 else 2) = 3;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_7: bool = bool_literal true
  // CHECK:STDOUT:   if %.loc[[@LINE-2]]_7 br !if.expr.then.loc[[@LINE-2]] else br !if.expr.else.loc[[@LINE-2]]
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.then.loc[[@LINE-4]]:
  // CHECK:STDOUT:   %.loc[[@LINE-5]]_17: i32 = int_literal 1
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-6]](%.loc[[@LINE-6]]_17)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else.loc[[@LINE-8]]:
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_24: i32 = int_literal 2
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-10]](%.loc[[@LINE-10]]_24)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result.loc[[@LINE-12]]:
  // CHECK:STDOUT:   %.loc[[@LINE-13]]_4: i32 = block_arg !if.expr.result.loc[[@LINE-13]]
  // CHECK:STDOUT:   %.loc[[@LINE-14]]_29: i32 = int_literal 3
  // CHECK:STDOUT:   assign %.loc[[@LINE-15]]_4, %.loc[[@LINE-15]]_29
  // CHECK:STDOUT:   %a: ref i32 = var "a"

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assigment_to_non_assignable.carbon:[[@LINE+3]]:25: Expression is not assignable.
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:                         ^
  (if true then a else a) = 10;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_7: bool = bool_literal true
  // CHECK:STDOUT:   if %.loc[[@LINE-2]]_7 br !if.expr.then.loc[[@LINE-2]] else br !if.expr.else.loc[[@LINE-2]]
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.then.loc[[@LINE-4]]:
  // CHECK:STDOUT:   %.loc[[@LINE-9]]_7.1: i32 = bind_value %a
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-6]](%.loc[[@LINE-10]]_7.1)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.else.loc[[@LINE-8]]:
  // CHECK:STDOUT:   %.loc[[@LINE-13]]_7.2: i32 = bind_value %a
  // CHECK:STDOUT:   br !if.expr.result.loc[[@LINE-10]](%.loc[[@LINE-14]]_7.2)
  // CHECK:STDOUT:
  // CHECK:STDOUT: !if.expr.result.loc[[@LINE-12]]:
  // CHECK:STDOUT:   %.loc[[@LINE-13]]_4: i32 = block_arg !if.expr.result.loc[[@LINE-13]]
  // CHECK:STDOUT:   %.loc[[@LINE-14]]_29: i32 = int_literal 10
  // CHECK:STDOUT:   assign %.loc[[@LINE-15]]_4, %.loc[[@LINE-15]]_29
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
