// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:42: ERROR: Duplicated field name `abc` in struct type literal.
// CHECK:STDERR: fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};
// CHECK:STDERR:                                          ^~~
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:22: Field with the same name here.
// CHECK:STDERR: fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};
// CHECK:STDERR:                      ^~~
fn F() -> {.d: i32, .abc: i32, .e: i32, .abc: i32, .f: i32};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:19: ERROR: Duplicated field name `a` in struct type literal.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:10: Field with the same name here.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:          ^
let v: {.a: i32, .a: i32} = {.a = 1};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:26: ERROR: Duplicated field name `def` in struct literal.
// CHECK:STDERR: let w: i32 = {.def = 1, .def = 2}.def;
// CHECK:STDERR:                          ^~~
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:16: Field with the same name here.
// CHECK:STDERR: let w: i32 = {.def = 1, .def = 2}.def;
// CHECK:STDERR:                ^~~
let w: i32 = {.def = 1, .def = 2}.def;

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:30: ERROR: Duplicated field name `a` in struct literal.
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                              ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:22: Field with the same name here.
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                      ^
var x: {.a: i32} = {.a = 1, .a = 2};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:39: ERROR: Duplicated field name `b` in struct literal.
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                                       ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:31: Field with the same name here.
// CHECK:STDERR: var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};
// CHECK:STDERR:                               ^
var y: {.b: i32, .c: i32} = {.b = 3, .b = 4};

// CHECK:STDOUT: --- fail_duplicate_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type {.b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .x = %x, .y = %y} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %.loc21_35: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc21_36: {.a: i32} = struct_literal (%.loc21_35)
// CHECK:STDOUT:   %v: <error> = bind_name v, <error>
// CHECK:STDOUT:   %.loc29_22: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc29_32: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %w: i32 = bind_name w, <error>
// CHECK:STDOUT:   %.loc37_16: type = struct_type {.a: i32} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref {.a: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc37_26: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc37_34: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   %.loc45_25: type = struct_type {.b: i32, .c: i32} [template = constants.%.4]
// CHECK:STDOUT:   %y.var: ref {.b: i32, .c: i32} = var y
// CHECK:STDOUT:   %y: ref {.b: i32, .c: i32} = bind_name y, %y.var
// CHECK:STDOUT:   %.loc45_35: i32 = int_literal 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc45_43: i32 = int_literal 4 [template = constants.%.7]
// CHECK:STDOUT:   assign %y.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> <error>;
// CHECK:STDOUT:
