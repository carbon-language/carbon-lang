// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:13: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:13: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:21: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:                     ^~~~~~~
// CHECK:STDERR:
var x: () = if true then () else ();

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:14: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:14: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:23: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:                       ^~~~~~~
// CHECK:STDERR:
var x2: () = if false then () else ();

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:15: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:15: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~~~~~~~
// CHECK:STDERR:
var y: bool = true or false;

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+7]]:16: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+3]]:16: ERROR: Semantics TODO: `Control flow expressions are currently only supported inside functions.`.
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~~~~~~
var y2: bool = false or true;

// CHECK:STDOUT: --- fail_todo_control_flow_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %.3: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .x2 = %x2
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .y2 = %y2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc19_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc19_9.2: type = converted %.loc19_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref () = var x
// CHECK:STDOUT:   %x: ref () = bind_name x, %x.var
// CHECK:STDOUT:   %.loc33_10.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc33_10.2: type = converted %.loc33_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x2.var: ref () = var x2
// CHECK:STDOUT:   %x2: ref () = bind_name x2, %x2.var
// CHECK:STDOUT:   %y.var: ref bool = var y
// CHECK:STDOUT:   %y: ref bool = bind_name y, %y.var
// CHECK:STDOUT:   %y2.var: ref bool = var y2
// CHECK:STDOUT:   %y2: ref bool = bind_name y2, %y2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc19 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT: }
// CHECK:STDOUT:
