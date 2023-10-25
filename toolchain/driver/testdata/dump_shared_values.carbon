// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --phase=lex --dump-shared-values %s
//
// AUTOUPDATE

// The value 8 is significant because it can show as negative due to APInt.
var int1: i32 = 1;
var int2: i32 = 8;
var real1: f64 = 1.0;
var real2: f64 = 0.8e8;
var real3: f64 = 0.8e9;
var str1: String = "abc";
var str2: String = "ab'\"c";

// CHECK:STDOUT: shared_values:
// CHECK:STDOUT:   - integers:
// CHECK:STDOUT:       - 32
// CHECK:STDOUT:       - 1
// CHECK:STDOUT:       - 32
// CHECK:STDOUT:       - 8
// CHECK:STDOUT:       - 64
// CHECK:STDOUT:       - 64
// CHECK:STDOUT:       - 64
// CHECK:STDOUT:     reals:
// CHECK:STDOUT:       - 10*10^-1
// CHECK:STDOUT:       - 8*10^7
// CHECK:STDOUT:       - 8*10^8
// CHECK:STDOUT:     strings:
// CHECK:STDOUT:       - "int1"
// CHECK:STDOUT:       - "int2"
// CHECK:STDOUT:       - "real1"
// CHECK:STDOUT:       - "real2"
// CHECK:STDOUT:       - "real3"
// CHECK:STDOUT:       - "str1"
// CHECK:STDOUT:       - "abc"
// CHECK:STDOUT:       - "str2"
// CHECK:STDOUT:       - "ab'\"c"
