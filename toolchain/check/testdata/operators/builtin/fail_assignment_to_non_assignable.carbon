// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32;

fn Main() {
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   1 = 2;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  1 = 2;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   F() = 1;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  F() = 1;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (1, 2) = (3, 4);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (1, 2) = (3, 4);
  var n: i32 = 0;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (n, n) = (1, 2);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  (n, n) = (1, 2);
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   i32 = i32*;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  i32 = i32*;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  {.x = 1, .y = 2} = {.x = 3, .y = 4};
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+4]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (if true then 1 else 2) = 3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (if true then 1 else 2) = 3;

  // Under #911, if expressions are never reference expressions.
  var a: i32;
  // CHECK:STDERR: fail_assignment_to_non_assignable.carbon:[[@LINE+3]]:3: ERROR: Expression is not assignable.
  // CHECK:STDERR:   (if true then a else a) = 10;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~
  (if true then a else a) = 10;
}

// CHECK:STDOUT: --- fail_assignment_to_non_assignable.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.6: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.7: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.8: (i32, i32) = tuple_value (%.1, %.2) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.11: type = struct_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type {.x: i32, .y: i32} [template]
// CHECK:STDOUT:   %.13: {.x: i32, .y: i32} = struct_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.14: {.x: i32, .y: i32} = struct_value (%.1, %.2) [template]
// CHECK:STDOUT:   %.15: bool = bool_literal true [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .Main = %Main
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_3: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_7: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   assign %.loc14_3, %.loc14_7
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc19_4: init i32 = call %F.ref()
// CHECK:STDOUT:   %.loc19_9: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   assign %.loc19_4, %.loc19_9
// CHECK:STDOUT:   %.loc24_4: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc24_7: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc24_8.1: (i32, i32) = tuple_literal (%.loc24_4, %.loc24_7)
// CHECK:STDOUT:   %.loc24_13: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc24_16: i32 = int_literal 4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc24_17.1: (i32, i32) = tuple_literal (%.loc24_13, %.loc24_16)
// CHECK:STDOUT:   %.loc24_17.2: i32 = tuple_access %.loc24_8.1, element0
// CHECK:STDOUT:   %.loc24_17.3: init i32 = initialize_from %.loc24_13 to %.loc24_17.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc24_17.4: i32 = tuple_access %.loc24_8.1, element1
// CHECK:STDOUT:   %.loc24_17.5: init i32 = initialize_from %.loc24_16 to %.loc24_17.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc24_17.6: init (i32, i32) = tuple_init (%.loc24_17.3, %.loc24_17.5) to %.loc24_8.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc24_17.7: init (i32, i32) = converted %.loc24_17.1, %.loc24_17.6 [template = constants.%.7]
// CHECK:STDOUT:   assign %.loc24_8.1, %.loc24_17.7
// CHECK:STDOUT:   %.loc24_8.2: (i32, i32) = tuple_value (%.loc24_4, %.loc24_7) [template = constants.%.8]
// CHECK:STDOUT:   %.loc24_8.3: (i32, i32) = converted %.loc24_8.1, %.loc24_8.2 [template = constants.%.8]
// CHECK:STDOUT:   %n.var: ref i32 = var n
// CHECK:STDOUT:   %n: ref i32 = bind_name n, %n.var
// CHECK:STDOUT:   %.loc25: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:   assign %n.var, %.loc25
// CHECK:STDOUT:   %n.ref.loc30_4: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %n.ref.loc30_7: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc30_8.1: (i32, i32) = tuple_literal (%n.ref.loc30_4, %n.ref.loc30_7)
// CHECK:STDOUT:   %.loc30_13: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc30_16: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_17.1: (i32, i32) = tuple_literal (%.loc30_13, %.loc30_16)
// CHECK:STDOUT:   %.loc30_17.2: i32 = tuple_access %.loc30_8.1, element0
// CHECK:STDOUT:   %.loc30_17.3: init i32 = initialize_from %.loc30_13 to %.loc30_17.2 [template = constants.%.1]
// CHECK:STDOUT:   %.loc30_17.4: i32 = tuple_access %.loc30_8.1, element1
// CHECK:STDOUT:   %.loc30_17.5: init i32 = initialize_from %.loc30_16 to %.loc30_17.4 [template = constants.%.2]
// CHECK:STDOUT:   %.loc30_17.6: init (i32, i32) = tuple_init (%.loc30_17.3, %.loc30_17.5) to %.loc30_8.1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc30_17.7: init (i32, i32) = converted %.loc30_17.1, %.loc30_17.6 [template = constants.%.8]
// CHECK:STDOUT:   assign %.loc30_8.1, %.loc30_17.7
// CHECK:STDOUT:   %.loc30_4: i32 = bind_value %n.ref.loc30_4
// CHECK:STDOUT:   %.loc30_7: i32 = bind_value %n.ref.loc30_7
// CHECK:STDOUT:   %.loc30_8.2: (i32, i32) = tuple_value (%.loc30_4, %.loc30_7)
// CHECK:STDOUT:   %.loc30_8.3: (i32, i32) = converted %.loc30_8.1, %.loc30_8.2
// CHECK:STDOUT:   %.loc35: type = ptr_type i32 [template = constants.%.10]
// CHECK:STDOUT:   assign i32, %.loc35
// CHECK:STDOUT:   %.loc40_9: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc40_17: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc40_18.1: {.x: i32, .y: i32} = struct_literal (%.loc40_9, %.loc40_17)
// CHECK:STDOUT:   %.loc40_28: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc40_36: i32 = int_literal 4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc40_37.1: {.x: i32, .y: i32} = struct_literal (%.loc40_28, %.loc40_36)
// CHECK:STDOUT:   %.loc40_37.2: i32 = struct_access %.loc40_18.1, element0
// CHECK:STDOUT:   %.loc40_37.3: init i32 = initialize_from %.loc40_28 to %.loc40_37.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc40_37.4: i32 = struct_access %.loc40_18.1, element1
// CHECK:STDOUT:   %.loc40_37.5: init i32 = initialize_from %.loc40_36 to %.loc40_37.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc40_37.6: init {.x: i32, .y: i32} = struct_init (%.loc40_37.3, %.loc40_37.5) to %.loc40_18.1 [template = constants.%.13]
// CHECK:STDOUT:   %.loc40_37.7: init {.x: i32, .y: i32} = converted %.loc40_37.1, %.loc40_37.6 [template = constants.%.13]
// CHECK:STDOUT:   assign %.loc40_18.1, %.loc40_37.7
// CHECK:STDOUT:   %.loc40_18.2: {.x: i32, .y: i32} = struct_value (%.loc40_9, %.loc40_17) [template = constants.%.14]
// CHECK:STDOUT:   %.loc40_18.3: {.x: i32, .y: i32} = converted %.loc40_18.1, %.loc40_18.2 [template = constants.%.14]
// CHECK:STDOUT:   %.loc45_7: bool = bool_literal true [template = constants.%.15]
// CHECK:STDOUT:   if %.loc45_7 br !if.expr.then.loc45 else br !if.expr.else.loc45
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc45:
// CHECK:STDOUT:   %.loc45_17: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc45(%.loc45_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc45:
// CHECK:STDOUT:   %.loc45_24: i32 = int_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   br !if.expr.result.loc45(%.loc45_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc45:
// CHECK:STDOUT:   %.loc45_4: i32 = block_arg !if.expr.result.loc45
// CHECK:STDOUT:   %.loc45_29: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   assign %.loc45_4, %.loc45_29
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc52_7: bool = bool_literal true [template = constants.%.15]
// CHECK:STDOUT:   if %.loc52_7 br !if.expr.then.loc52 else br !if.expr.else.loc52
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc52:
// CHECK:STDOUT:   %a.ref.loc52_17: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc52_17: i32 = bind_value %a.ref.loc52_17
// CHECK:STDOUT:   br !if.expr.result.loc52(%.loc52_17)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc52:
// CHECK:STDOUT:   %a.ref.loc52_24: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc52_24: i32 = bind_value %a.ref.loc52_24
// CHECK:STDOUT:   br !if.expr.result.loc52(%.loc52_24)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc52:
// CHECK:STDOUT:   %.loc52_4: i32 = block_arg !if.expr.result.loc52
// CHECK:STDOUT:   %.loc52_29: i32 = int_literal 10 [template = constants.%.16]
// CHECK:STDOUT:   assign %.loc52_4, %.loc52_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
