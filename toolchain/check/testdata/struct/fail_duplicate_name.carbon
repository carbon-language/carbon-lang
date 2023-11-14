// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:13: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: fn F() -> {.b: i32, .b: i32};
// CHECK:STDERR:             ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:13: Member with the same name here.
// CHECK:STDERR: fn F() -> {.b: i32, .b: i32};
// CHECK:STDERR:             ^
fn F() -> {.b: i32, .b: i32};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+15]]:10: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+12]]:10: Member with the same name here.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+9]]:33: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:33: Member with the same name here.
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:36: ERROR: Cannot initialize struct with 2 field(s) from struct with 1 field(s).
// CHECK:STDERR: let v: {.a: i32, .a: i32} = {.a = 1};
// CHECK:STDERR:                                    ^
let v: {.a: i32, .a: i32} = {.a = 1};

// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+6]]:18: ERROR: Member of struct literal with a duplicated name.
// CHECK:STDERR: let w: i32 = {.a = 1, .a = 2}.a;
// CHECK:STDERR:                  ^
// CHECK:STDERR: fail_duplicate_name.carbon:[[@LINE+3]]:18: Member with the same name here.
// CHECK:STDERR: let w: i32 = {.a = 1, .a = 2}.a;
// CHECK:STDERR:                  ^
let w: i32 = {.a = 1, .a = 2}.a;

// var x: {.a: i32} = {.a = 1, .a = 2};

// var y: {.b: i32, .c: i32} = {.a = 3, .a = 4};

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7: type = ptr_type {.b: i32, .b: i32}
// CHECK:STDOUT:   %.loc12_25: type = ptr_type {.a: i32, .a: i32}
// CHECK:STDOUT:   %.loc12_36: type = struct_type {.a: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_duplicate_name.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %.loc12_25: type = struct_type {.a: i32, .a: i32}
// CHECK:STDOUT:   %.loc12_35: i32 = int_literal 1
// CHECK:STDOUT:   %.loc12_36: {.a: i32} = struct_literal (%.loc12_35)
// CHECK:STDOUT:   %v: {.a: i32, .a: i32} = bind_name v, <error>
// CHECK:STDOUT:   %.loc14_20: i32 = int_literal 1
// CHECK:STDOUT:   %.loc14_28: i32 = int_literal 2
// CHECK:STDOUT:   %.loc14_29.1: {.a: i32, .a: i32} = struct_literal (%.loc14_20, %.loc14_28)
// CHECK:STDOUT:   %.loc14_29.2: {.a: i32, .a: i32} = struct_value (%.loc14_20, %.loc14_28)
// CHECK:STDOUT:   %.loc14_29.3: {.a: i32, .a: i32} = converted %.loc14_29.1, %.loc14_29.2
// CHECK:STDOUT:   %.loc14_30: i32 = struct_access %.loc14_29.3, member0
// CHECK:STDOUT:   %w: i32 = bind_name w, %.loc14_30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: {.b: i32, .b: i32};
