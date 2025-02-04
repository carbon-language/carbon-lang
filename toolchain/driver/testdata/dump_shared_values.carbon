// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --phase=lex --dump-shared-values %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/dump_shared_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/dump_shared_values.carbon

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
// CHECK:STDOUT:   ints:            {}
// CHECK:STDOUT:   reals:
// CHECK:STDOUT:     real0:           10*10^-1
// CHECK:STDOUT:     real1:           8*10^7
// CHECK:STDOUT:     real2:           8*10^8
// CHECK:STDOUT:   identifiers:
// CHECK:STDOUT:     identifier0:     int1
// CHECK:STDOUT:     identifier1:     int2
// CHECK:STDOUT:     identifier2:     real1
// CHECK:STDOUT:     identifier3:     real2
// CHECK:STDOUT:     identifier4:     real3
// CHECK:STDOUT:     identifier5:     str1
// CHECK:STDOUT:     identifier6:     str2
// CHECK:STDOUT:   strings:
// CHECK:STDOUT:     string0:         abc
// CHECK:STDOUT:     string1:         ab'"c
// CHECK:STDOUT: ...
