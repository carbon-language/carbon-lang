// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/import.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- api.carbon

library "api";

fn A();
fn B(b: i32) -> i32;
fn C(c: (i32,)) -> {.c: i32};
extern fn D();

namespace NS;
fn NS.E();

// --- extern_api.carbon

library "extern_api";

extern fn A();
extern fn B(b: i32) -> i32;
extern fn C(c: (i32,)) -> {.c: i32};
extern fn D();

namespace NS;
extern fn NS.E();

// ============================================================================
// Test files
// ============================================================================

// --- basics.carbon

library "basics";

import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- redecl_api.carbon

library "redecl_api";

import library "api";

extern fn A();
extern fn B(b: i32) -> i32;
extern fn C(c: (i32,)) -> {.c: i32};
extern fn D();
extern fn NS.E();

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- redecl_extern_api.carbon

library "redecl_extern_api";

import library "extern_api";

extern fn A();
extern fn B(b: i32) -> i32;
extern fn C(c: (i32,)) -> {.c: i32};
extern fn D();
extern fn NS.E();

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- fail_todo_merge.carbon

library "merge";

import library "api";
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE+65]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE+52]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:5:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE-20]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:5:1: Name is previously declared here.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE+39]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:6:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE-33]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:6:1: Name is previously declared here.
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE+26]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:7:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE-46]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:7:1: Name is previously declared here.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:10:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge.carbon:[[@LINE-59]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:10:1: Name is previously declared here.
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
import library "extern_api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- fail_todo_merge_reverse.carbon

library "merge_reverse";

import library "extern_api";
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE+65]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE+52]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:5:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE-20]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:5:1: Name is previously declared here.
// CHECK:STDERR: extern fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE+39]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:6:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE-33]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:6:1: Name is previously declared here.
// CHECK:STDERR: extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE+26]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:7:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE-46]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:7:1: Name is previously declared here.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:10:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_todo_merge_reverse.carbon:[[@LINE-59]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:10:1: Name is previously declared here.
// CHECK:STDERR: extern fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- decl_after_use.carbon

library "decl_after_use";

import library "extern_api";

var a: () = A();

fn A();

// --- fail_redecl_mismatch_after_use.carbon

library "redecl_mismatch_after_use";

import library "extern_api";

var a: () = A();

// CHECK:STDERR: fail_redecl_mismatch_after_use.carbon:[[@LINE+10]]:1: ERROR: Function redeclaration differs because return type is `i32`.
// CHECK:STDERR: fn A() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_mismatch_after_use.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:4:1: Previously declared with no return type.
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn A() -> i32;

// --- todo_fail_extern_after_use.carbon

library "extern_after_use";

import library "api";

var a: () = A();

extern fn A();

// --- unloaded.carbon

library "unloaded";

import library "api";

// --- unloaded_extern.carbon

library "unloaded_extern";

import library "extern_api";

// --- fail_todo_loaded_merge.carbon

library "loaded_merge";

import library "api";
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE+64]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE-7]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE+51]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:5:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE-20]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:5:1: Name is previously declared here.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE+38]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:6:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE-33]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:6:1: Name is previously declared here.
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE+25]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:7:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE-46]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:7:1: Name is previously declared here.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:10:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_loaded_merge.carbon:[[@LINE-59]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:10:1: Name is previously declared here.
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
import library "extern_api";

// CHECK:STDOUT: --- api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: <type of D> = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: <type of E> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: <type of A> = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %import_ref.1: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %B.decl: <type of B> = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc5_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_9.1: type = value_of_initializer %int.make_type_32.loc5_9 [template = i32]
// CHECK:STDOUT:     %.loc5_9.2: type = converted %int.make_type_32.loc5_9, %.loc5_9.1 [template = i32]
// CHECK:STDOUT:     %b.loc5_6.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_6.1
// CHECK:STDOUT:     %int.make_type_32.loc5_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_17.1: type = value_of_initializer %int.make_type_32.loc5_17 [template = i32]
// CHECK:STDOUT:     %.loc5_17.2: type = converted %int.make_type_32.loc5_17, %.loc5_17.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.4: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %C.decl: <type of C> = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc6_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_14.1: (type,) = tuple_literal (%int.make_type_32.loc6_10)
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_32.loc6_10 [template = i32]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_32.loc6_10, %.loc6_14.2 [template = i32]
// CHECK:STDOUT:     %.loc6_14.4: type = converted %.loc6_14.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc6_6.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_6.1
// CHECK:STDOUT:     %int.make_type_32.loc6_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_25.1: type = value_of_initializer %int.make_type_32.loc6_25 [template = i32]
// CHECK:STDOUT:     %.loc6_25.2: type = converted %int.make_type_32.loc6_25, %.loc6_25.1 [template = i32]
// CHECK:STDOUT:     %.loc6_28: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     @C.%return: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: <type of D> = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: <type of E> = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: <type of D> = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: <type of E> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: <type of A> = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %import_ref.1: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %B.decl: <type of B> = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc5_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_16.1: type = value_of_initializer %int.make_type_32.loc5_16 [template = i32]
// CHECK:STDOUT:     %.loc5_16.2: type = converted %int.make_type_32.loc5_16, %.loc5_16.1 [template = i32]
// CHECK:STDOUT:     %b.loc5_13.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_13.1
// CHECK:STDOUT:     %int.make_type_32.loc5_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_24.1: type = value_of_initializer %int.make_type_32.loc5_24 [template = i32]
// CHECK:STDOUT:     %.loc5_24.2: type = converted %int.make_type_32.loc5_24, %.loc5_24.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.3: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.4: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %C.decl: <type of C> = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc6_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_21.1: (type,) = tuple_literal (%int.make_type_32.loc6_17)
// CHECK:STDOUT:     %.loc6_21.2: type = value_of_initializer %int.make_type_32.loc6_17 [template = i32]
// CHECK:STDOUT:     %.loc6_21.3: type = converted %int.make_type_32.loc6_17, %.loc6_21.2 [template = i32]
// CHECK:STDOUT:     %.loc6_21.4: type = converted %.loc6_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc6_13.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_13.1
// CHECK:STDOUT:     %int.make_type_32.loc6_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_32.1: type = value_of_initializer %int.make_type_32.loc6_32 [template = i32]
// CHECK:STDOUT:     %.loc6_32.2: type = converted %int.make_type_32.loc6_32, %.loc6_32.1 [template = i32]
// CHECK:STDOUT:     %.loc6_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     @C.%return: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: <type of D> = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: <type of E> = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: <type of D> = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: <type of E> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: <type of B> = import_ref ir1, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: <type of C> = import_ref ir1, inst+43, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: <type of D> = import_ref ir1, inst+46, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <type of E> = import_ref ir1, inst+50, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.7: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_32.loc7, %.loc7_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.8: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_32.loc8, %.loc8_13.1 [template = i32]
// CHECK:STDOUT:   %.loc8_16: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc9_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc9_9.2: type = converted %.loc9_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc10_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <type of B> = name_ref B, file.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc7)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <type of C> = name_ref C, file.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %.loc8_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25: (i32,) = tuple_literal (%.loc8_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc8_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_21: (i32,) = converted %.loc8_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc8_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <type of D> = name_ref D, file.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <type of E> = name_ref E, file.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: <type of D> = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: <type of E> = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b.loc13
// CHECK:STDOUT:     .c = %c.loc14
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: <type of B> = import_ref ir1, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: <type of C> = import_ref ir1, inst+43, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: <type of D> = import_ref ir1, inst+46, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <type of E> = import_ref ir1, inst+50, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: <type of A> = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %import_ref.7: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.8: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %B.decl: <type of B> = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc7_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_16.1: type = value_of_initializer %int.make_type_32.loc7_16 [template = i32]
// CHECK:STDOUT:     %.loc7_16.2: type = converted %int.make_type_32.loc7_16, %.loc7_16.1 [template = i32]
// CHECK:STDOUT:     %b.loc7_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc7_13.2: i32 = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %int.make_type_32.loc7_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_24.1: type = value_of_initializer %int.make_type_32.loc7_24 [template = i32]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %int.make_type_32.loc7_24, %.loc7_24.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.10: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %C.decl: <type of C> = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc8_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_21.1: (type,) = tuple_literal (%int.make_type_32.loc8_17)
// CHECK:STDOUT:     %.loc8_21.2: type = value_of_initializer %int.make_type_32.loc8_17 [template = i32]
// CHECK:STDOUT:     %.loc8_21.3: type = converted %int.make_type_32.loc8_17, %.loc8_21.2 [template = i32]
// CHECK:STDOUT:     %.loc8_21.4: type = converted %.loc8_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc8_13.1: (i32,) = param c
// CHECK:STDOUT:     %c.loc8_13.2: (i32,) = bind_name c, %c.loc8_13.1
// CHECK:STDOUT:     %int.make_type_32.loc8_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_32.1: type = value_of_initializer %int.make_type_32.loc8_32 [template = i32]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %int.make_type_32.loc8_32, %.loc8_32.1 [template = i32]
// CHECK:STDOUT:     %.loc8_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     %return.var.loc8: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: <type of D> = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %E.decl: <type of E> = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.11: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_32.loc13, %.loc13_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc13: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.12: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_13.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_13.2: type = converted %int.make_type_32.loc14, %.loc14_13.1 [template = i32]
// CHECK:STDOUT:   %.loc14_16: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c.loc14: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc15_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: type = converted %.loc15_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc16_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <type of B> = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc13)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <type of C> = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc14_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: (i32,) = tuple_literal (%.loc14_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc14_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_21: (i32,) = converted %.loc14_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc14_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <type of D> = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <type of E> = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: <type of D> = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: <type of E> = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b.loc13
// CHECK:STDOUT:     .c = %c.loc14
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: <type of B> = import_ref ir1, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: <type of C> = import_ref ir1, inst+43, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: <type of D> = import_ref ir1, inst+46, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <type of E> = import_ref ir1, inst+50, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: <type of A> = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %import_ref.7: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.8: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %B.decl: <type of B> = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc7_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_16.1: type = value_of_initializer %int.make_type_32.loc7_16 [template = i32]
// CHECK:STDOUT:     %.loc7_16.2: type = converted %int.make_type_32.loc7_16, %.loc7_16.1 [template = i32]
// CHECK:STDOUT:     %b.loc7_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc7_13.2: i32 = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %int.make_type_32.loc7_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_24.1: type = value_of_initializer %int.make_type_32.loc7_24 [template = i32]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %int.make_type_32.loc7_24, %.loc7_24.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.9: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.10: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %C.decl: <type of C> = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc8_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_21.1: (type,) = tuple_literal (%int.make_type_32.loc8_17)
// CHECK:STDOUT:     %.loc8_21.2: type = value_of_initializer %int.make_type_32.loc8_17 [template = i32]
// CHECK:STDOUT:     %.loc8_21.3: type = converted %int.make_type_32.loc8_17, %.loc8_21.2 [template = i32]
// CHECK:STDOUT:     %.loc8_21.4: type = converted %.loc8_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc8_13.1: (i32,) = param c
// CHECK:STDOUT:     %c.loc8_13.2: (i32,) = bind_name c, %c.loc8_13.1
// CHECK:STDOUT:     %int.make_type_32.loc8_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_32.1: type = value_of_initializer %int.make_type_32.loc8_32 [template = i32]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %int.make_type_32.loc8_32, %.loc8_32.1 [template = i32]
// CHECK:STDOUT:     %.loc8_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     %return.var.loc8: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: <type of D> = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %E.decl: <type of E> = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.11: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_32.loc13, %.loc13_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc13: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.12: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_13.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_13.2: type = converted %int.make_type_32.loc14, %.loc14_13.1 [template = i32]
// CHECK:STDOUT:   %.loc14_16: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c.loc14: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc15_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: type = converted %.loc15_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc16_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <type of B> = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc13)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <type of C> = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc14_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: (i32,) = tuple_literal (%.loc14_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc14_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_21: (i32,) = converted %.loc14_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc14_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <type of D> = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <type of E> = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: <type of D> = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: <type of E> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: <type of B> = import_ref ir1, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: <type of C> = import_ref ir1, inst+43, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: <type of D> = import_ref ir1, inst+46, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <type of E> = import_ref ir1, inst+50, loaded [template = constants.%E]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir2, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir2, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref ir2, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc72_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc72_9.2: type = converted %.loc72_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.12: <type of Int32> = import_ref ir5, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc73: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc73_8.1: type = value_of_initializer %int.make_type_32.loc73 [template = i32]
// CHECK:STDOUT:   %.loc73_8.2: type = converted %int.make_type_32.loc73, %.loc73_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.13: <type of Int32> = import_ref ir5, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc74: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc74_13.1: type = value_of_initializer %int.make_type_32.loc74 [template = i32]
// CHECK:STDOUT:   %.loc74_13.2: type = converted %int.make_type_32.loc74, %.loc74_13.1 [template = i32]
// CHECK:STDOUT:   %.loc74_16: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc75_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc75_9.2: type = converted %.loc75_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc76_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc76_9.2: type = converted %.loc76_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <type of B> = name_ref B, file.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %.loc73: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc73)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <type of C> = name_ref C, file.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %.loc74_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc74_25: (i32,) = tuple_literal (%.loc74_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc74_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc74_21: (i32,) = converted %.loc74_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc74_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <type of D> = name_ref D, file.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <type of E> = name_ref E, file.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_merge_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: <type of B> = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: <type of C> = struct_value () [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: <type of D> = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: <type of E> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: <type of B> = import_ref ir1, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: <type of C> = import_ref ir1, inst+43, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: <type of D> = import_ref ir1, inst+46, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <type of E> = import_ref ir1, inst+50, loaded [template = constants.%E]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir2, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir2, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref ir2, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc72_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc72_9.2: type = converted %.loc72_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.12: <type of Int32> = import_ref ir5, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc73: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc73_8.1: type = value_of_initializer %int.make_type_32.loc73 [template = i32]
// CHECK:STDOUT:   %.loc73_8.2: type = converted %int.make_type_32.loc73, %.loc73_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %import_ref.13: <type of Int32> = import_ref ir5, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %int.make_type_32.loc74: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc74_13.1: type = value_of_initializer %int.make_type_32.loc74 [template = i32]
// CHECK:STDOUT:   %.loc74_13.2: type = converted %int.make_type_32.loc74, %.loc74_13.1 [template = i32]
// CHECK:STDOUT:   %.loc74_16: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc75_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc75_9.2: type = converted %.loc75_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc76_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc76_9.2: type = converted %.loc76_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <type of B> = name_ref B, file.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %.loc73: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc73)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <type of C> = name_ref C, file.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %.loc74_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc74_25: (i32,) = tuple_literal (%.loc74_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc74_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc74_21: (i32,) = converted %.loc74_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc74_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <type of D> = name_ref D, file.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <type of E> = name_ref E, file.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A.decl: <type of A> = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_mismatch_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %import_ref.7: <type of Int32> = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %.decl: <type of <invalid>> = fn_decl @.1 [template = constants.%.2] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc18_11.2: type = converted %int.make_type_32, %.loc18_11.1 [template = i32]
// CHECK:STDOUT:     @.1.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_extern_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: <type of A> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <type of A> = import_ref ir1, inst+2, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A.decl: <type of A> = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <type of A> = name_ref A, file.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unloaded_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_loaded_merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+49, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+50, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir2, inst+43, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir2, inst+46, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref ir2, inst+50, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
