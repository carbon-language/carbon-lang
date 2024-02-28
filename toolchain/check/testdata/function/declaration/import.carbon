// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

fn A();
fn B(b: i32) -> i32;
fn C(c: (i32,)) -> {.c: i32};

// --- b.carbon

library "b" api;

import library "a";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template]
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template]
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: (i32,)) -> {.c: i32};
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
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
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .A = %import_ref.1
// CHECK:STDOUT:     .B = %import_ref.2
// CHECK:STDOUT:     .C = %import_ref.3
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %import_ref.1: <function> = import_ref ir1, inst+1, used [template = imports.%A]
// CHECK:STDOUT:   %import_ref.2: <function> = import_ref ir1, inst+5, used [template = imports.%B]
// CHECK:STDOUT:   %import_ref.3: <function> = import_ref ir1, inst+17, used [template = imports.%C]
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
