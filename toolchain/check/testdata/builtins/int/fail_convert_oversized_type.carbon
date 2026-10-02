// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/fail_convert_oversized_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/fail_convert_oversized_type.carbon

// --- fail_width_not_multiple_of_8.carbon

library "[[@TEST_NAME]]";

// Initializing a binding whose type is an invalid integer type must not crash
// constant evaluation (#7278). The type is diagnosed; the conversion of the
// initializer should then produce an error value, not a CHECK failure.
// CHECK:STDERR: fail_width_not_multiple_of_8.carbon:[[@LINE+5]]:10-17: error: bit width of integer type literal must be a multiple of 8; use `Core.Int(8388609)` instead [IntWidthNotMultipleOf8]
// CHECK:STDERR:
// CHECK:STDERR: fail_width_not_multiple_of_8.carbon:[[@LINE+3]]:10-17: error: binding pattern has incomplete type `i8388609` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:18:9-18: note: integer type width of 8388609 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:
let bad: i8388609 = 0;

// --- fail_width_too_large.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_width_too_large.carbon:[[@LINE+3]]:10-18: error: binding pattern has incomplete type `i16777216` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:18:9-18: note: integer type width of 16777216 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:
let bad: i16777216 = 0;

// --- fail_width_too_large_core_int.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_width_too_large_core_int.carbon:[[@LINE+3]]:10-26: error: binding pattern has incomplete type `i8388609` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: {{.*}}/prelude/types/int.carbon:18:9-18: note: integer type width of 8388609 is greater than the maximum supported width of 8388608 [IntWidthTooLarge]
// CHECK:STDERR:
let bad: Core.Int(8388609) = 0;
