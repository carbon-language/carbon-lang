// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:                     ^~~~~~~
// CHECK:STDERR:
var x: () = if true then () else ();

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:23: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:                       ^~~~~~~
// CHECK:STDERR:
var x2: () = if false then () else ();

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:15: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:15: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~~~~~~~
// CHECK:STDERR:
var y: bool = true or false;

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+7]]:16: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+3]]:16: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~~~~~~
var y2: bool = false or true;

// CHECK:STDOUT: --- fail_todo_control_flow_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .x2 = %x2
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .y2 = %y2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc23_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_9.2: type = converted %.loc23_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT:   %.loc37_10.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc37_10.2: type = converted %.loc37_10.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %x2.var: ref %empty_tuple.type = var x2
// CHECK:STDOUT:   %x2: ref %empty_tuple.type = bind_name x2, %x2.var
// CHECK:STDOUT:   %bool.make_type.loc47: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc47_8.1: type = value_of_initializer %bool.make_type.loc47 [template = bool]
// CHECK:STDOUT:   %.loc47_8.2: type = converted %bool.make_type.loc47, %.loc47_8.1 [template = bool]
// CHECK:STDOUT:   %y.var: ref bool = var y
// CHECK:STDOUT:   %y: ref bool = bind_name y, %y.var
// CHECK:STDOUT:   %bool.make_type.loc56: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc56_9.1: type = value_of_initializer %bool.make_type.loc56 [template = bool]
// CHECK:STDOUT:   %.loc56_9.2: type = converted %bool.make_type.loc56, %.loc56_9.1 [template = bool]
// CHECK:STDOUT:   %y2.var: ref bool = var y2
// CHECK:STDOUT:   %y2: ref bool = bind_name y2, %y2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc23 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT: }
// CHECK:STDOUT:
