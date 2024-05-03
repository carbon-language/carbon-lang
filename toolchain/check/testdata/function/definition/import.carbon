// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- fns.carbon

library "fns" api;

fn A() {}
fn B(b: i32) -> i32 { return b; }
fn C(c: (i32,)) -> {.c: i32} { return {.c = c[0]}; }
fn D();

// --- extern.carbon

library "extern" api;

extern fn A();

// ============================================================================
// Test files
// ============================================================================

// --- basics.carbon

library "basics" api;

import library "fns";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));

// --- fail_def_ownership.carbon

library "def_ownership" api;

import library "fns";

// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE+10]]:1: ERROR: Only one library can declare `fn A` without `extern`.
// CHECK:STDERR: fn A() {};
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "fns";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fns.carbon:4:1: Previously declared here.
// CHECK:STDERR: fn A() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn A() {};
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE+10]]:1: ERROR: Only one library can declare `fn B` without `extern`.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE-16]]:1: In import.
// CHECK:STDERR: import library "fns";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: fns.carbon:5:1: Previously declared here.
// CHECK:STDERR: fn B(b: i32) -> i32 { return b; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(b: i32) -> i32;

// --- redecl_then_def.carbon

library "redecl_then_def" api;

import library "extern";

fn A();
fn A() {}

// --- fail_mix_extern_decl.carbon

library "mix_extern_decl" api;

import library "fns";

extern fn D();
// CHECK:STDERR: fail_mix_extern_decl.carbon:[[@LINE+6]]:1: ERROR: Redeclarations of `fn D` in the same library must match use of `extern`.
// CHECK:STDERR: fn D() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_mix_extern_decl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
fn D() {}

// CHECK:STDOUT: --- fns.carbon
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
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.4: D = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   return %b.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: (i32,)) -> {.c: i32} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: (i32,) = name_ref c, %c
// CHECK:STDOUT:   %.loc6_47: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc6_48: i32 = tuple_index %c.ref, %.loc6_47
// CHECK:STDOUT:   %.loc6_49: {.c: i32} = struct_literal (%.loc6_48)
// CHECK:STDOUT:   %struct: {.c: i32} = struct_value (%.loc6_48)
// CHECK:STDOUT:   %.loc6_50: {.c: i32} = converted %.loc6_49, %struct
// CHECK:STDOUT:   return %.loc6_50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: B = import_ref ir1, inst+10, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: C = import_ref ir1, inst+26, loaded [template = constants.%struct.3]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+38, unloaded
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: (i32,)) -> {.c: i32};
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
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_def_ownership.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: A = import_ref ir1, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: B = import_ref ir1, inst+10, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+38, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %b.loc27_6.1: i32 = param b
// CHECK:STDOUT:     %b.loc27_6.2: i32 = bind_name b, %b.loc27_6.1
// CHECK:STDOUT:     %return.var: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_then_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: A = import_ref ir1, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl.loc6: A = fn_decl @A [template = constants.%struct] {}
// CHECK:STDOUT:   %A.decl.loc7: A = fn_decl @A [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mix_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: D = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .D = %D.decl.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+10, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+26, unloaded
// CHECK:STDOUT:   %import_ref.4: D = import_ref ir1, inst+38, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %D.decl.loc6: D = fn_decl @D [template = constants.%struct] {}
// CHECK:STDOUT:   %D.decl.loc13: D = fn_decl @D [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
