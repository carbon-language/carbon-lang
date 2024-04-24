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

// --- merge.carbon

library "merge" api;

import library "api";
import library "extern_api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- merge_reverse.carbon

library "merge_reverse" api;

import library "extern_api";
import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- fail_decl_after_use.carbon

library "decl_after_use" api;

import library "extern_api";

var a: () = A();

// CHECK:STDERR: fail_decl_after_use.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of imported entity that was previously used.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_decl_after_use.carbon:[[@LINE-5]]:13: Import used here.
// CHECK:STDERR: var a: () = A();
// CHECK:STDERR:             ^
// CHECK:STDERR:
fn A();

// --- fail_decl_after_use_two_errors.carbon

library "decl_after_use_two_errors" api;

import library "extern_api";

var a: () = A();

// CHECK:STDERR: fail_decl_after_use_two_errors.carbon:[[@LINE+17]]:1: ERROR: Redeclaration of imported entity that was previously used.
// CHECK:STDERR: fn A() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_after_use_two_errors.carbon:[[@LINE-5]]:13: Import used here.
// CHECK:STDERR: var a: () = A();
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_decl_after_use_two_errors.carbon:[[@LINE+10]]:1: ERROR: Function redeclaration differs because return type is `i32`.
// CHECK:STDERR: fn A() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_after_use_two_errors.carbon:[[@LINE-14]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:4:1: Previously declared with no return type.
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn A() -> i32;

// --- fail_extern_after_use.carbon

library "extern_after_use" api;

import library "api";

var a: () = A();

// CHECK:STDERR: fail_extern_after_use.carbon:[[@LINE+6]]:1: ERROR: Redeclaration of imported entity that was previously used.
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_after_use.carbon:[[@LINE-5]]:13: Import used here.
// CHECK:STDERR: var a: () = A();
// CHECK:STDERR:             ^
extern fn A();

// --- unloaded.carbon

library "unloaded" api;

import library "api";

// --- unloaded_extern.carbon

library "unloaded_extern" api;

import library "extern_api";

// --- loaded_merge.carbon

library "loaded_merge" api;

import library "api";
import library "extern_api";

// CHECK:STDOUT: --- api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %b.loc5_6.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_6.1
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %.loc6_14.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc6_14.2: type = converted %.loc6_14.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %c.loc6_6.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_6.1
// CHECK:STDOUT:     %.loc6_28: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:     @C.%return: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E: <function> = fn_decl @E [template] {}
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
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %b.loc5_13.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_13.1
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %.loc6_21.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc6_21.2: type = converted %.loc6_21.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %c.loc6_13.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_13.1
// CHECK:STDOUT:     %.loc6_35: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:     @C.%return: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E: <function> = fn_decl @E [template] {}
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
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.5: (i32,) = tuple_value (%.2) [template]
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
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loc_15 [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, loc_24 [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, loc_39 [template = imports.%C]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+19, loc_53 [template = imports.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+21, loc_65 [template = imports.%E]
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
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%import_ref.2 [template = imports.%B]
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc7)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%import_ref.3 [template = imports.%C]
// CHECK:STDOUT:   %.loc8_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25.1: (i32,) = tuple_literal (%.loc8_23)
// CHECK:STDOUT:   %.loc8_25.2: (i32,) = tuple_value (%.loc8_23) [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_21: (i32,) = converted %.loc8_25.1, %.loc8_25.2 [template = constants.%.5]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc8_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <function> = name_ref E, file.%import_ref.6 [template = imports.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: (i32,) = tuple_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b.loc13
// CHECK:STDOUT:     .c = %c.loc14
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loaded [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, loaded [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, loaded [template = imports.%C]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+19, loaded [template = imports.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+21, loaded [template = imports.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %b.loc7_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc7_13.2: i32 = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %.loc8_21.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc8_21.2: type = converted %.loc8_21.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %c.loc8_13.1: (i32,) = param c
// CHECK:STDOUT:     %c.loc8_13.2: (i32,) = bind_name c, %c.loc8_13.1
// CHECK:STDOUT:     %.loc8_35: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:     %return.var.loc8: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %E: <function> = fn_decl @E [template] {}
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc13: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c.loc14: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc15_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: type = converted %.loc15_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc16_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%.4 [template = constants.%.4]
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
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%B [template = file.%B]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc13)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%C [template = file.%C]
// CHECK:STDOUT:   %.loc14_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25.1: (i32,) = tuple_literal (%.loc14_23)
// CHECK:STDOUT:   %.loc14_25.2: (i32,) = tuple_value (%.loc14_23) [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_21: (i32,) = converted %.loc14_25.1, %.loc14_25.2 [template = constants.%.6]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc14_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%D [template = file.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <function> = name_ref E, file.%E [template = file.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: (i32,) = tuple_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b.loc13
// CHECK:STDOUT:     .c = %c.loc14
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loaded [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, loaded [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, loaded [template = imports.%C]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+19, loaded [template = imports.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+21, loaded [template = imports.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %b.loc7_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc7_13.2: i32 = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %.loc8_21.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc8_21.2: type = converted %.loc8_21.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %c.loc8_13.1: (i32,) = param c
// CHECK:STDOUT:     %c.loc8_13.2: (i32,) = bind_name c, %c.loc8_13.1
// CHECK:STDOUT:     %.loc8_35: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:     %return.var.loc8: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %E: <function> = fn_decl @E [template] {}
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc13: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc14: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c.loc14: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc15_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: type = converted %.loc15_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc16_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%.4 [template = constants.%.4]
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
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%A [template = file.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%B [template = file.%B]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc13)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%C [template = file.%C]
// CHECK:STDOUT:   %.loc14_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25.1: (i32,) = tuple_literal (%.loc14_23)
// CHECK:STDOUT:   %.loc14_25.2: (i32,) = tuple_value (%.loc14_23) [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_21: (i32,) = converted %.loc14_25.1, %.loc14_25.2 [template = constants.%.6]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc14_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%D [template = file.%D]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <function> = name_ref E, file.%E [template = file.%E]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.2: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: (i32,) = tuple_value (%.4) [template]
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
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loc_19 [template = imports.%A.1]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, loc_28 [template = imports.%B.1]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, loc_43 [template = imports.%C.1]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+19, loc_57 [template = imports.%D.1]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+21, loc_69 [template = imports.%E.1]
// CHECK:STDOUT:   %import_ref.7: <function> = import_ref ir2, inst+2, loaded [template = imports.%A.2]
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir2, inst+6, loaded [template = imports.%B.2]
// CHECK:STDOUT:   %import_ref.9: <function> = import_ref ir2, inst+18, loaded [template = imports.%C.2]
// CHECK:STDOUT:   %import_ref.10: <function> = import_ref ir2, inst+19, loaded [template = imports.%D.2]
// CHECK:STDOUT:   %import_ref.11: <function> = import_ref ir2, inst+21, loaded [template = imports.%E.2]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc9: type = struct_type {.c: i32} [template = constants.%.2]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc10_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc11_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.2();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B.1(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.2(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C.1(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.2(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D.1();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D.2();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A.1]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%import_ref.2 [template = imports.%B.1]
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc8)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%import_ref.3 [template = imports.%C.1]
// CHECK:STDOUT:   %.loc9_23: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_25.1: (i32,) = tuple_literal (%.loc9_23)
// CHECK:STDOUT:   %.loc9_25.2: (i32,) = tuple_value (%.loc9_23) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_21: (i32,) = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%.5]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc9_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D.1]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <function> = name_ref E, file.%import_ref.6 [template = imports.%E.1]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- merge_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.2: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: (i32,) = tuple_value (%.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:     .A = imports.%A.2
// CHECK:STDOUT:     .B = imports.%B.2
// CHECK:STDOUT:     .C = imports.%C.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loaded [template = imports.%A.1]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, loaded [template = imports.%B.1]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, loaded [template = imports.%C.1]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+19, loc_57 [template = imports.%D.1]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = imports.%E.2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+21, loaded [template = imports.%E.1]
// CHECK:STDOUT:   %import_ref.7: <function> = import_ref ir2, inst+2, loaded [template = imports.%A.2]
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir2, inst+6, loaded [template = imports.%B.2]
// CHECK:STDOUT:   %import_ref.9: <function> = import_ref ir2, inst+18, loaded [template = imports.%C.2]
// CHECK:STDOUT:   %import_ref.10: <function> = import_ref ir2, inst+19, loaded [template = imports.%D.2]
// CHECK:STDOUT:   %import_ref.11: <function> = import_ref ir2, inst+21, loaded [template = imports.%E.2]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc7_9.2: type = converted %.loc7_9.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc9: type = struct_type {.c: i32} [template = constants.%.2]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc10_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc11_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %e.var: ref () = var e
// CHECK:STDOUT:   %e: ref () = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.1(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.2(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.1(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.2(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D.1();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, imports.%A.2 [template = imports.%A.2]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, imports.%B.2 [template = imports.%B.2]
// CHECK:STDOUT:   %.loc8: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc8)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, imports.%C.2 [template = imports.%C.2]
// CHECK:STDOUT:   %.loc9_23: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_25.1: (i32,) = tuple_literal (%.loc9_23)
// CHECK:STDOUT:   %.loc9_25.2: (i32,) = tuple_value (%.loc9_23) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_21: (i32,) = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%.5]
// CHECK:STDOUT:   %C.call: init {.c: i32} = call %C.ref(%.loc9_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D.1]
// CHECK:STDOUT:   %D.call: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, file.%NS [template = file.%NS]
// CHECK:STDOUT:   %E.ref: <function> = name_ref E, imports.%E.2 [template = imports.%E.2]
// CHECK:STDOUT:   %E.call: init () = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
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
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loc_15 [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+6, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+18, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_decl_after_use_two_errors.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
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
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loc_15 [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+6, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+18, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %.loc25: <function> = fn_decl @.1 [template] {
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
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A]
// CHECK:STDOUT:   %A.call: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_after_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
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
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loc_15 [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+6, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+18, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc6_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A]
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
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+6, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+18, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+21, unloaded
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
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+6, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+18, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+19, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref ir1, inst+21, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- loaded_merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.2: type = struct_type {.c: i32} [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+2, loaded [template = imports.%A.1]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, loaded [template = imports.%B.1]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+18, loaded [template = imports.%C.1]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+19, loaded [template = imports.%D.1]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref ir1, inst+20, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir1, inst+21, loaded [template = imports.%E.1]
// CHECK:STDOUT:   %import_ref.7: <function> = import_ref ir2, inst+2, loaded [template = imports.%A.2]
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir2, inst+6, loaded [template = imports.%B.2]
// CHECK:STDOUT:   %import_ref.9: <function> = import_ref ir2, inst+18, loaded [template = imports.%C.2]
// CHECK:STDOUT:   %import_ref.10: <function> = import_ref ir2, inst+19, loaded [template = imports.%D.2]
// CHECK:STDOUT:   %import_ref.11: <function> = import_ref ir2, inst+21, loaded [template = imports.%E.2]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.2();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B.1(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.2(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C.1(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.2(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D.1();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D.2();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.2();
// CHECK:STDOUT:
