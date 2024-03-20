// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

library "a" api;

alias a = i32;
alias b = a;
alias c = b;
alias d = c;

// --- b.carbon

library "b" api;

import library "a";

// Access imports in reverse order of export.
var d_val: d = 1;
var c_val: c = d_val;
var b_val: b = c_val;
var a_val: a = b_val;
var i32_val: i32 = a_val;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: type = bind_alias a, i32 [template = i32]
// CHECK:STDOUT:   %a.ref: type = name_ref a, %a [template = i32]
// CHECK:STDOUT:   %b: type = bind_alias b, %a [template = i32]
// CHECK:STDOUT:   %b.ref: type = name_ref b, %b [template = i32]
// CHECK:STDOUT:   %c: type = bind_alias c, %b [template = i32]
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [template = i32]
// CHECK:STDOUT:   %d: type = bind_alias d, %c [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %import_ref.1
// CHECK:STDOUT:     .b = %import_ref.2
// CHECK:STDOUT:     .c = %import_ref.3
// CHECK:STDOUT:     .d = %import_ref.4
// CHECK:STDOUT:     .d_val = %d_val
// CHECK:STDOUT:     .c_val = %c_val
// CHECK:STDOUT:     .b_val = %b_val
// CHECK:STDOUT:     .a_val = %a_val
// CHECK:STDOUT:     .i32_val = %i32_val
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir1, inst+1, used [template = i32]
// CHECK:STDOUT:   %import_ref.2: type = import_ref ir1, inst+3, used [template = i32]
// CHECK:STDOUT:   %import_ref.3: type = import_ref ir1, inst+5, used [template = i32]
// CHECK:STDOUT:   %import_ref.4: type = import_ref ir1, inst+7, used [template = i32]
// CHECK:STDOUT:   %d.ref: type = name_ref d, %import_ref.4 [template = i32]
// CHECK:STDOUT:   %d_val.var: ref i32 = var d_val
// CHECK:STDOUT:   %d_val: ref i32 = bind_name d_val, %d_val.var
// CHECK:STDOUT:   %c.ref: type = name_ref c, %import_ref.3 [template = i32]
// CHECK:STDOUT:   %c_val.var: ref i32 = var c_val
// CHECK:STDOUT:   %c_val: ref i32 = bind_name c_val, %c_val.var
// CHECK:STDOUT:   %b.ref: type = name_ref b, %import_ref.2 [template = i32]
// CHECK:STDOUT:   %b_val.var: ref i32 = var b_val
// CHECK:STDOUT:   %b_val: ref i32 = bind_name b_val, %b_val.var
// CHECK:STDOUT:   %a.ref: type = name_ref a, %import_ref.1 [template = i32]
// CHECK:STDOUT:   %a_val.var: ref i32 = var a_val
// CHECK:STDOUT:   %a_val: ref i32 = bind_name a_val, %a_val.var
// CHECK:STDOUT:   %i32_val.var: ref i32 = var i32_val
// CHECK:STDOUT:   %i32_val: ref i32 = bind_name i32_val, %i32_val.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   assign file.%d_val.var, %.loc7
// CHECK:STDOUT:   %d_val.ref: ref i32 = name_ref d_val, file.%d_val
// CHECK:STDOUT:   %.loc8: i32 = bind_value %d_val.ref
// CHECK:STDOUT:   assign file.%c_val.var, %.loc8
// CHECK:STDOUT:   %c_val.ref: ref i32 = name_ref c_val, file.%c_val
// CHECK:STDOUT:   %.loc9: i32 = bind_value %c_val.ref
// CHECK:STDOUT:   assign file.%b_val.var, %.loc9
// CHECK:STDOUT:   %b_val.ref: ref i32 = name_ref b_val, file.%b_val
// CHECK:STDOUT:   %.loc10: i32 = bind_value %b_val.ref
// CHECK:STDOUT:   assign file.%a_val.var, %.loc10
// CHECK:STDOUT:   %a_val.ref: ref i32 = name_ref a_val, file.%a_val
// CHECK:STDOUT:   %.loc11: i32 = bind_value %a_val.ref
// CHECK:STDOUT:   assign file.%i32_val.var, %.loc11
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
