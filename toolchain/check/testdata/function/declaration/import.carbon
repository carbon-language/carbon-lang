// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- api.carbon

library "api" api;

fn A();
fn B(b: i32) -> i32;
fn C(c: (i32,)) -> {.c: i32};
extern fn D();

namespace NS;
fn NS.E();

// --- extern_api.carbon

library "extern_api" api;

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

library "basics" api;

import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- redecl_api.carbon

library "redecl_api" api;

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

library "redecl_extern_api" api;

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

library "merge" api;

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

library "merge_reverse" api;

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

library "decl_after_use" api;

import library "extern_api";

var a: () = A();

fn A();

// --- fail_redecl_mismatch_after_use.carbon

library "redecl_mismatch_after_use" api;

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

library "extern_after_use" api;

import library "api";

var a: () = A();

extern fn A();

// --- unloaded.carbon

library "unloaded" api;

import library "api";

// --- unloaded_extern.carbon

library "unloaded_extern" api;

import library "extern_api";

// --- fail_todo_loaded_merge.carbon

library "loaded_merge" api;

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
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
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
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %b.loc5_6.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_6.1
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     %.loc6_14.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc6_14.2: type = converted %.loc6_14.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc6_6.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_6.1
// CHECK:STDOUT:     %.loc6_28: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     @C.%return: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: D = fn_decl @D [template = constants.%struct.4] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: E = fn_decl @E [template = constants.%struct.5] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
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
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
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
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %b.loc5_13.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_13.1
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     %.loc6_21.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc6_21.2: type = converted %.loc6_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc6_13.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_13.1
// CHECK:STDOUT:     %.loc6_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     @C.%return: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: D = fn_decl @D [template = constants.%struct.4] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: E = fn_decl @E [template = constants.%struct.5] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
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
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: B = import_ref ir1, inst+9, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: C = import_ref ir1, inst+23, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.4: D = import_ref ir1, inst+26, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: E = import_ref ir1, inst+30, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc8: type = struct_type {.c: i32} [template = constants.%.3]
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
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%import_ref.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%import_ref.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc7)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%import_ref.3 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc8_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25: (i32,) = tuple_literal (%.loc8_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc8_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_21: (i32,) = converted %.loc8_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc8_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: D = name_ref D, file.%import_ref.4 [template = constants.%struct.4]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: E = name_ref E, file.%import_ref.6 [template = constants.%struct.5]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: B = import_ref ir1, inst+9, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: C = import_ref ir1, inst+23, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.4: D = import_ref ir1, inst+26, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: E = import_ref ir1, inst+30, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %b.loc7_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc7_13.2: i32 = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     %.loc8_21.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc8_21.2: type = converted %.loc8_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc8_13.1: (i32,) = param c
// CHECK:STDOUT:     %c.loc8_13.2: (i32,) = bind_name c, %c.loc8_13.1
// CHECK:STDOUT:     %.loc8_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     %return.var.loc8: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: D = fn_decl @D [template = constants.%struct.4] {}
// CHECK:STDOUT:   %E.decl: E = fn_decl @E [template = constants.%struct.5] {}
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc13: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14: type = struct_type {.c: i32} [template = constants.%.4]
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
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc13)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%C.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc14_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: (i32,) = tuple_literal (%.loc14_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc14_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_21: (i32,) = converted %.loc14_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc14_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: D = name_ref D, file.%D.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: E = name_ref E, file.%E.decl [template = constants.%struct.5]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: B = import_ref ir1, inst+9, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: C = import_ref ir1, inst+23, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.4: D = import_ref ir1, inst+26, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: E = import_ref ir1, inst+30, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %b.loc7_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc7_13.2: i32 = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     %.loc8_21.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc8_21.2: type = converted %.loc8_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc8_13.1: (i32,) = param c
// CHECK:STDOUT:     %c.loc8_13.2: (i32,) = bind_name c, %c.loc8_13.1
// CHECK:STDOUT:     %.loc8_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     %return.var.loc8: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: D = fn_decl @D [template = constants.%struct.4] {}
// CHECK:STDOUT:   %E.decl: E = fn_decl @E [template = constants.%struct.5] {}
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc13: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14: type = struct_type {.c: i32} [template = constants.%.4]
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
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%A.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%B.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc13)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%C.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc14_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: (i32,) = tuple_literal (%.loc14_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc14_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_21: (i32,) = converted %.loc14_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc14_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: D = name_ref D, file.%D.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: E = name_ref E, file.%E.decl [template = constants.%struct.5]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: B = import_ref ir1, inst+9, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: C = import_ref ir1, inst+23, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.4: D = import_ref ir1, inst+26, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: E = import_ref ir1, inst+30, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir2, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir2, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref ir2, inst+30, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc72_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc72_9.2: type = converted %.loc72_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc74: type = struct_type {.c: i32} [template = constants.%.3]
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
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%import_ref.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%import_ref.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc73: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc73)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%import_ref.3 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc74_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc74_25: (i32,) = tuple_literal (%.loc74_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc74_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc74_21: (i32,) = converted %.loc74_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc74_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: D = name_ref D, file.%import_ref.4 [template = constants.%struct.4]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: E = name_ref E, file.%import_ref.6 [template = constants.%struct.5]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_merge_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.5: E = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: B = import_ref ir1, inst+9, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: C = import_ref ir1, inst+23, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.4: D = import_ref ir1, inst+26, loaded [template = constants.%struct.4]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: E = import_ref ir1, inst+30, loaded [template = constants.%struct.5]
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir2, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir2, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref ir2, inst+30, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc72_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc72_9.2: type = converted %.loc72_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc74: type = struct_type {.c: i32} [template = constants.%.3]
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
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%import_ref.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: B = name_ref B, file.%import_ref.2 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc73: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc73)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: C = name_ref C, file.%import_ref.3 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc74_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc74_25: (i32,) = tuple_literal (%.loc74_23)
// CHECK:STDOUT:   %tuple: (i32,) = tuple_value (%.loc74_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc74_21: (i32,) = converted %.loc74_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc74_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: D = name_ref D, file.%import_ref.4 [template = constants.%struct.4]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: E = name_ref E, file.%import_ref.6 [template = constants.%struct.5]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- decl_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct: A = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+30, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%import_ref.1 [template = constants.%struct]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_mismatch_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %.2: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+30, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     @.1.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%import_ref.1 [template = constants.%struct.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_extern_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %struct: A = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+30, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: A = name_ref A, file.%import_ref.1 [template = constants.%struct]
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
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+30, unloaded
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
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+30, unloaded
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
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+29, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+30, unloaded
// CHECK:STDOUT:   %import_ref.7 = import_ref ir2, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref ir2, inst+9, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref ir2, inst+23, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref ir2, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref ir2, inst+30, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
