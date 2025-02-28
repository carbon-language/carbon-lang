// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/printing_digit_padding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/printing_digit_padding.carbon
// CHECK:STDOUT: - filename: printing_digit_padding.carbon
// CHECK:STDOUT:   tokens:

;
// CHECK:STDOUT:   - { index: 1, kind:      "Semi", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: ";", has_leading_space: true }

// Test digit padding with values of 9, 10, and 11.
        ;;;
        // CHECK:STDOUT:   - { index: 2, kind:      "Semi", line: {{ *}}[[@LINE-1]], column:   9, indent: 9, spelling: ";", has_leading_space: true }
        // CHECK:STDOUT:   - { index: 3, kind:      "Semi", line: {{ *}}[[@LINE-2]], column:  10, indent: 9, spelling: ";" }
        // CHECK:STDOUT:   - { index: 4, kind:      "Semi", line: {{ *}}[[@LINE-3]], column:  11, indent: 9, spelling: ";" }
