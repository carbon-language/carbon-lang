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

// --- extern_api.carbon

library "extern_api" api;

extern fn A();
extern fn B(b: i32) -> i32;
extern fn C(c: (i32,)) -> {.c: i32};
extern fn D();

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

// --- redecl_api.carbon

library "redecl_api" api;

import library "api";

extern fn A();
extern fn B(b: i32) -> i32;
extern fn C(c: (i32,)) -> {.c: i32};
extern fn D();

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();

// --- redecl_extern_api.carbon

library "redecl_extern_api" api;

import library "extern_api";

extern fn A();
extern fn B(b: i32) -> i32;
extern fn C(c: (i32,)) -> {.c: i32};
extern fn D();

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();

// --- merge.carbon

library "merge" api;

import library "api";
import library "extern_api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();

// --- merge_reverse.carbon

library "merge_reverse" api;

import library "extern_api";
import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();

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
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %b.loc5_6.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_6.1
// CHECK:STDOUT:     %return.var.loc5: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %.loc6_14.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc6_14.2: type = converted %.loc6_14.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %c.loc6_6.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_6.1
// CHECK:STDOUT:     %.loc6_28: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:     %return.var.loc6: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D: <function> = fn_decl @D [template] {}
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
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %b.loc5_13.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_13.1
// CHECK:STDOUT:     %return.var.loc5: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %.loc6_21.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:     %.loc6_21.2: type = converted %.loc6_21.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     %c.loc6_13.1: (i32,) = param c
// CHECK:STDOUT:     @C.%c: (i32,) = bind_name c, %c.loc6_13.1
// CHECK:STDOUT:     %.loc6_35: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:     %return.var.loc6: ref {.c: i32} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D: <function> = fn_decl @D [template] {}
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+5, used [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+17, used [template = imports.%C]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+18, used [template = imports.%D]
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A]
// CHECK:STDOUT:   %.loc6: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %.loc6
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%import_ref.2 [template = imports.%B]
// CHECK:STDOUT:   %.loc7_16: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_15: init i32 = call %B.ref(%.loc7_16)
// CHECK:STDOUT:   assign file.%b.var, %.loc7_15
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%import_ref.3 [template = imports.%C]
// CHECK:STDOUT:   %.loc8_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25.1: (i32,) = tuple_literal (%.loc8_23)
// CHECK:STDOUT:   %.loc8_25.2: (i32,) = tuple_value (%.loc8_23) [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_25.3: (i32,) = converted %.loc8_25.1, %.loc8_25.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_21: init {.c: i32} = call %C.ref(%.loc8_25.3)
// CHECK:STDOUT:   assign file.%c.var, %.loc8_21
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D]
// CHECK:STDOUT:   %.loc9: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %.loc9
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
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b.loc12
// CHECK:STDOUT:     .c = %c.loc13
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+5, used [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+17, used [template = imports.%C]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+18, used [template = imports.%D]
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
// CHECK:STDOUT:   %.loc11_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc12: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc13: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c.loc13: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc14_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc14_9.2: type = converted %.loc14_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A]
// CHECK:STDOUT:   %.loc11: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %.loc11
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%import_ref.2 [template = imports.%B]
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_15: init i32 = call %B.ref(%.loc12_16)
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%import_ref.3 [template = imports.%C]
// CHECK:STDOUT:   %.loc13_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_25.1: (i32,) = tuple_literal (%.loc13_23)
// CHECK:STDOUT:   %.loc13_25.2: (i32,) = tuple_value (%.loc13_23) [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_25.3: (i32,) = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_21: init {.c: i32} = call %C.ref(%.loc13_25.3)
// CHECK:STDOUT:   assign file.%c.var, %.loc13_21
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D]
// CHECK:STDOUT:   %.loc14: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %.loc14
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
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b.loc12
// CHECK:STDOUT:     .c = %c.loc13
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+5, used [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+17, used [template = imports.%C]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+18, used [template = imports.%D]
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
// CHECK:STDOUT:   %.loc11_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc12: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc13: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref {.c: i32} = var c
// CHECK:STDOUT:   %c.loc13: ref {.c: i32} = bind_name c, %c.var
// CHECK:STDOUT:   %.loc14_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc14_9.2: type = converted %.loc14_9.1, constants.%.4 [template = constants.%.4]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A]
// CHECK:STDOUT:   %.loc11: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %.loc11
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%import_ref.2 [template = imports.%B]
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_15: init i32 = call %B.ref(%.loc12_16)
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%import_ref.3 [template = imports.%C]
// CHECK:STDOUT:   %.loc13_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_25.1: (i32,) = tuple_literal (%.loc13_23)
// CHECK:STDOUT:   %.loc13_25.2: (i32,) = tuple_value (%.loc13_23) [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_25.3: (i32,) = converted %.loc13_25.1, %.loc13_25.2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_21: init {.c: i32} = call %C.ref(%.loc13_25.3)
// CHECK:STDOUT:   assign file.%c.var, %.loc13_21
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D]
// CHECK:STDOUT:   %.loc14: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %.loc14
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A.1]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+5, used [template = imports.%B.1]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+17, used [template = imports.%C.1]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+18, used [template = imports.%D.1]
// CHECK:STDOUT:   %import_ref.5: <function> = import_ref ir2, inst+1, used [template = imports.%A.2]
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir2, inst+5, used [template = imports.%B.2]
// CHECK:STDOUT:   %import_ref.7: <function> = import_ref ir2, inst+17, used [template = imports.%C.2]
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir2, inst+18, used [template = imports.%D.2]
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A.1]
// CHECK:STDOUT:   %.loc7: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %.loc7
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%import_ref.2 [template = imports.%B.1]
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_15: init i32 = call %B.ref(%.loc8_16)
// CHECK:STDOUT:   assign file.%b.var, %.loc8_15
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%import_ref.3 [template = imports.%C.1]
// CHECK:STDOUT:   %.loc9_23: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_25.1: (i32,) = tuple_literal (%.loc9_23)
// CHECK:STDOUT:   %.loc9_25.2: (i32,) = tuple_value (%.loc9_23) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_25.3: (i32,) = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_21: init {.c: i32} = call %C.ref(%.loc9_25.3)
// CHECK:STDOUT:   assign file.%c.var, %.loc9_21
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D.1]
// CHECK:STDOUT:   %.loc10: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %.loc10
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
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A.1]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+5, used [template = imports.%B.1]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+17, used [template = imports.%C.1]
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+18, used [template = imports.%D.1]
// CHECK:STDOUT:   %import_ref.5: <function> = import_ref ir2, inst+1, used [template = imports.%A.2]
// CHECK:STDOUT:   %import_ref.6: <function> = import_ref ir2, inst+5, used [template = imports.%B.2]
// CHECK:STDOUT:   %import_ref.7: <function> = import_ref ir2, inst+17, used [template = imports.%C.2]
// CHECK:STDOUT:   %import_ref.8: <function> = import_ref ir2, inst+18, used [template = imports.%D.2]
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
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: <function> = name_ref A, file.%import_ref.1 [template = imports.%A.1]
// CHECK:STDOUT:   %.loc7: init () = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %.loc7
// CHECK:STDOUT:   %B.ref: <function> = name_ref B, file.%import_ref.2 [template = imports.%B.1]
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_15: init i32 = call %B.ref(%.loc8_16)
// CHECK:STDOUT:   assign file.%b.var, %.loc8_15
// CHECK:STDOUT:   %C.ref: <function> = name_ref C, file.%import_ref.3 [template = imports.%C.1]
// CHECK:STDOUT:   %.loc9_23: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_25.1: (i32,) = tuple_literal (%.loc9_23)
// CHECK:STDOUT:   %.loc9_25.2: (i32,) = tuple_value (%.loc9_23) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_25.3: (i32,) = converted %.loc9_25.1, %.loc9_25.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_21: init {.c: i32} = call %C.ref(%.loc9_25.3)
// CHECK:STDOUT:   assign file.%c.var, %.loc9_21
// CHECK:STDOUT:   %D.ref: <function> = name_ref D, file.%import_ref.4 [template = imports.%D.1]
// CHECK:STDOUT:   %.loc10: init () = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %.loc10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
