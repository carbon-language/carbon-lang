// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_todo_control_flow_init.carbon

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+16]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:21: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:                     ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:13: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x: () = if true then () else ();
// CHECK:STDERR:             ^~~~~~~
// CHECK:STDERR:
var x: () = if true then () else ();

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+16]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:23: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:                       ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var x2: () = if false then () else ();
// CHECK:STDERR:              ^~~~~~~~
// CHECK:STDERR:
var x2: () = if false then () else ();

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:15: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:15: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:15: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: bool = true or false;
// CHECK:STDERR:               ^~~~~~~~~~~~~
// CHECK:STDERR:
var y: bool = true or false;

// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+12]]:16: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+8]]:16: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_control_flow_init.carbon:[[@LINE+4]]:16: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y2: bool = false or true;
// CHECK:STDERR:                ^~~~~~~~~~~~~
// CHECK:STDERR:
var y2: bool = false or true;

// CHECK:STDOUT: --- fail_todo_control_flow_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .x2 = %x2
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .y2 = %y2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc27_1: %empty_tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc27_9.1: type = splice_block %.loc27_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc27_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc27_9.3: type = converted %.loc27_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x2.patt: %empty_tuple.type = binding_pattern x2
// CHECK:STDOUT:     %.loc45_1: %empty_tuple.type = var_pattern %x2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x2.var: ref %empty_tuple.type = var x2
// CHECK:STDOUT:   %.loc45_10.1: type = splice_block %.loc45_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc45_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc45_10.3: type = converted %.loc45_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x2: ref %empty_tuple.type = bind_name x2, %x2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: bool = binding_pattern y
// CHECK:STDOUT:     %.loc59_1: bool = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref bool = var y
// CHECK:STDOUT:   %.loc59_8.1: type = splice_block %.loc59_8.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc59: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc59_8.2: type = value_of_initializer %bool.make_type.loc59 [concrete = bool]
// CHECK:STDOUT:     %.loc59_8.3: type = converted %bool.make_type.loc59, %.loc59_8.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref bool = bind_name y, %y.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y2.patt: bool = binding_pattern y2
// CHECK:STDOUT:     %.loc73_1: bool = var_pattern %y2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y2.var: ref bool = var y2
// CHECK:STDOUT:   %.loc73_9.1: type = splice_block %.loc73_9.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc73: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc73_9.2: type = value_of_initializer %bool.make_type.loc73 [concrete = bool]
// CHECK:STDOUT:     %.loc73_9.3: type = converted %bool.make_type.loc73, %.loc73_9.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y2: ref bool = bind_name y2, %y2.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT: }
// CHECK:STDOUT:
