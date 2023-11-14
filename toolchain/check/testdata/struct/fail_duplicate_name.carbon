// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:22: ERROR: Member of struct type literal with a duplicated name `b`.
// CHECK:STDERR: fn F() -> {.b: i32, .b: i32};
// CHECK:STDERR:                      ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:13: Member with the same name here.
// CHECK:STDERR: fn F() -> {.b: i32, .b: i32};
// CHECK:STDERR:             ^
fn F() -> {.b: i32, .b: i32};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+9]]:19: ERROR: Member of struct type literal with a duplicated name `a`.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                   ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:10: Member with the same name here.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:36: ERROR: Cannot initialize struct with 2 field(s) from struct with 1 field(s).
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                                    ^
let v: {.a: i32, .a: i32} = {.a = 1};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:26: ERROR: Member of struct literal with a duplicated name `a`.
// CHECK:STDERR: let w: i32 = {.a = 1, .a = 2}.a;
// CHECK:STDERR:                          ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:18: Member with the same name here.
// CHECK:STDERR: let w: i32 = {.a = 1, .a = 2}.a;
// CHECK:STDERR:                  ^
let w: i32 = {.a = 1, .a = 2}.a;

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+9]]:32: ERROR: Member of struct literal with a duplicated name `a`.
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                                ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:24: Member with the same name here.
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                        ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:35: ERROR: Cannot initialize struct with 1 field(s) from struct with 2 field(s).
// CHECK:STDERR: var x: {.a: i32} = {.a = 1, .a = 2};
// CHECK:STDERR:                                   ^
var x: {.a: i32} = {.a = 1, .a = 2};

// var y: {.b: i32, .c: i32} = {.a = 3, .a = 4};

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc13: type = ptr_type {.b: i32, .b: i32}
// CHECK:STDOUT:   %.loc24_25: type = ptr_type {.a: i32, .a: i32}
// CHECK:STDOUT:   %.loc24_36: type = struct_type {.a: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_duplicate_name.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %.loc24_25: type = struct_type {.a: i32, .a: i32}
// CHECK:STDOUT:   %.loc24_35: i32 = int_literal 1
// CHECK:STDOUT:   %.loc24_36: {.a: i32} = struct_literal (%.loc24_35)
// CHECK:STDOUT:   %v: {.a: i32, .a: i32} = bind_name v, <error>
// CHECK:STDOUT:   %.loc32_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc32_28: i32 = int_literal 2
// CHECK:STDOUT:   %.loc32_29.1: {.a: i32, .a: i32} = struct_literal (%.loc32_20, %.loc32_28)
// CHECK:STDOUT:   %.loc32_29.2: {.a: i32, .a: i32} = struct_value (%.loc32_20, %.loc32_28)
// CHECK:STDOUT:   %.loc32_29.3: {.a: i32, .a: i32} = converted %.loc32_29.1, %.loc32_29.2
// CHECK:STDOUT:   %.loc32_30: i32 = struct_access %.loc32_29.3, member0
// CHECK:STDOUT:   %w: i32 = bind_name w, %.loc32_30
// CHECK:STDOUT:   %.loc43_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x.var: ref {.a: i32} = var x
// CHECK:STDOUT:   %x: ref {.a: i32} = bind_name x, %x.var
// CHECK:STDOUT:   %.loc43_26: i32 = int_literal 1
// CHECK:STDOUT:   %.loc43_34: i32 = int_literal 2
// CHECK:STDOUT:   %.loc43_35: {.a: i32, .a: i32} = struct_literal (%.loc43_26, %.loc43_34)
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: {.b: i32, .b: i32};
