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

// CHECK:STDOUT: ---
// CHECK:STDOUT: filename:        dump_shared_values.carbon
// CHECK:STDOUT: shared_values:
// CHECK:STDOUT:   ints:
// CHECK:STDOUT:     int0:            32
// CHECK:STDOUT:     int1:            1
// CHECK:STDOUT:     int2:            32
// CHECK:STDOUT:     int3:            8
// CHECK:STDOUT:     int4:            64
// CHECK:STDOUT:     int5:            64
// CHECK:STDOUT:     int6:            64
// CHECK:STDOUT:   reals:
// CHECK:STDOUT:     real0:           10*10^-1
// CHECK:STDOUT:     real1:           8*10^7
// CHECK:STDOUT:     real2:           8*10^8
// CHECK:STDOUT:   strings:
// CHECK:STDOUT:     str0:            int1
// CHECK:STDOUT:     str1:            int2
// CHECK:STDOUT:     str2:            real1
// CHECK:STDOUT:     str3:            real2
// CHECK:STDOUT:     str4:            real3
// CHECK:STDOUT:     str5:            str1
// CHECK:STDOUT:     str6:            abc
// CHECK:STDOUT:     str7:            str2
// CHECK:STDOUT:     str8:            'ab''"c'
// CHECK:STDOUT: ...
