// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/periods.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/periods.carbon
// CHECK:STDOUT: - filename: periods.carbon
// CHECK:STDOUT:   tokens:

. .. ... .... ...... ...expand
// CHECK:STDOUT:   - { index:  1, kind:             "Period", line: {{ *}}[[@LINE-1]], column:   1, indent: 1, spelling: ".", has_leading_space: true }
// CHECK:STDOUT:   - { index:  2, kind:             "Period", line: {{ *}}[[@LINE-2]], column:   3, indent: 1, spelling: ".", has_leading_space: true }
// CHECK:STDOUT:   - { index:  3, kind:             "Period", line: {{ *}}[[@LINE-3]], column:   4, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index:  4, kind: "PeriodPeriodPeriod", line: {{ *}}[[@LINE-4]], column:   6, indent: 1, spelling: "...", has_leading_space: true }
// CHECK:STDOUT:   - { index:  5, kind: "PeriodPeriodPeriod", line: {{ *}}[[@LINE-5]], column:  10, indent: 1, spelling: "...", has_leading_space: true }
// CHECK:STDOUT:   - { index:  6, kind:             "Period", line: {{ *}}[[@LINE-6]], column:  13, indent: 1, spelling: "." }
// CHECK:STDOUT:   - { index:  7, kind: "PeriodPeriodPeriod", line: {{ *}}[[@LINE-7]], column:  15, indent: 1, spelling: "...", has_leading_space: true }
// CHECK:STDOUT:   - { index:  8, kind: "PeriodPeriodPeriod", line: {{ *}}[[@LINE-8]], column:  18, indent: 1, spelling: "..." }
// CHECK:STDOUT:   - { index:  9, kind: "PeriodPeriodPeriod", line: {{ *}}[[@LINE-9]], column:  22, indent: 1, spelling: "...", has_leading_space: true }
// CHECK:STDOUT:   - { index: 10, kind:             "Expand", line: {{ *}}[[@LINE-10]], column:  25, indent: 1, spelling: "expand" }
