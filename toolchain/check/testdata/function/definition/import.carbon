// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// ============================================================================
// Setup files
// ============================================================================

// --- a.carbon

library "a" api;

fn A() {}
fn B(b: i32) -> i32 { return b; }
fn C(c: (i32,)) -> {.c: i32} { return {.c = c[0]}; }
fn D();

// ============================================================================
// Test files
// ============================================================================

// --- basics.carbon

library "basics" api;

import library "a";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));

// --- fail_def_ownership.carbon

library "def_ownership" api;

import library "a";

// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE+10]]:1: ERROR: Only one library can declare function A without `extern`.
// CHECK:STDERR: fn A() {};
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "a";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: a.carbon:4:1: Previously declared here.
// CHECK:STDERR: fn A() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn A() {};
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE+10]]:1: ERROR: Only one library can declare function B without `extern`.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_def_ownership.carbon:[[@LINE-16]]:1: In import.
// CHECK:STDERR: import library "a";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: a.carbon:5:1: Previously declared here.
// CHECK:STDERR: fn B(b: i32) -> i32 { return b; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(b: i32) -> i32;

// --- fail_mix_extern_decl.carbon

library "mix_extern_decl" api;

import library "a";

extern fn D();
// CHECK:STDERR: fail_mix_extern_decl.carbon:[[@LINE+9]]:1: ERROR: Only one library can declare function D without `extern`.
// CHECK:STDERR: fn D() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_mix_extern_decl.carbon:[[@LINE-6]]:1: In import.
// CHECK:STDERR: import library "a";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: a.carbon:7:1: Previously declared here.
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
fn D() {}

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
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
// CHECK:STDOUT:   %.loc6_47: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc6_48: i32 = tuple_index %c.ref, %.loc6_47
// CHECK:STDOUT:   %.loc6_49.1: {.c: i32} = struct_literal (%.loc6_48)
// CHECK:STDOUT:   %.loc6_49.2: {.c: i32} = struct_value (%.loc6_48)
// CHECK:STDOUT:   %.loc6_49.3: {.c: i32} = converted %.loc6_49.1, %.loc6_49.2
// CHECK:STDOUT:   return %.loc6_49.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D();
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, used [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+20, used [template = imports.%C]
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+30, unused
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
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_def_ownership.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+6, used [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+20, unused
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+30, unused
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
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
// CHECK:STDOUT: --- fail_mix_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .D = %import_ref.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1 = import_ref ir1, inst+1, unused
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+6, unused
// CHECK:STDOUT:   %import_ref.3 = import_ref ir1, inst+20, unused
// CHECK:STDOUT:   %import_ref.4: <function> = import_ref ir1, inst+30, used [template = imports.%D]
// CHECK:STDOUT:   %D.loc6: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %D.loc16: <function> = fn_decl @D [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
