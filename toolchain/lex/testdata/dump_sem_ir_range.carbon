// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lex/testdata/dump_sem_ir_range.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lex/testdata/dump_sem_ir_range.carbon

// --- full_file.carbon
// CHECK:STDOUT: - filename: full_file.carbon
// CHECK:STDOUT:   tokens:

//@dump-sem-ir-begin
a
// CHECK:STDOUT:   - { index: 1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "a", identifier: 0, has_leading_space: true }
b
// CHECK:STDOUT:   - { index: 2, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "b", identifier: 1, has_leading_space: true }
c
// CHECK:STDOUT:   - { index: 3, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "c", identifier: 2, has_leading_space: true }
// CHECK:STDOUT:   dump_sem_ir_ranges:
// CHECK:STDOUT:   - {begin: 1, end: 3}
//@dump-sem-ir-end

// --- multi_section.carbon
// CHECK:STDOUT: - filename: multi_section.carbon
// CHECK:STDOUT:   tokens:

a
// CHECK:STDOUT:   - { index: 1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "a", identifier: 0, has_leading_space: true }
//@dump-sem-ir-begin
b
// CHECK:STDOUT:   - { index: 2, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "b", identifier: 1, has_leading_space: true }
//@dump-sem-ir-end
c
// CHECK:STDOUT:   - { index: 3, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "c", identifier: 2, has_leading_space: true }
//@dump-sem-ir-begin
d
// CHECK:STDOUT:   - { index: 4, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "d", identifier: 3, has_leading_space: true }
//@dump-sem-ir-end
e
// CHECK:STDOUT:   - { index: 5, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "e", identifier: 4, has_leading_space: true }
// CHECK:STDOUT:   dump_sem_ir_ranges:
// CHECK:STDOUT:   - {begin: 2, end: 2}
// CHECK:STDOUT:   - {begin: 4, end: 4}

// --- compact.carbon
// CHECK:STDOUT: - filename: compact.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   dump_sem_ir_ranges:
// CHECK:STDOUT:   - {begin: 1, end: 0}
// CHECK:STDOUT:   - {begin: 1, end: 0}
// CHECK:STDOUT:   - {begin: 1, end: 0}
//@dump-sem-ir-begin
//@dump-sem-ir-end
//@dump-sem-ir-begin
//@dump-sem-ir-end
//@dump-sem-ir-begin
//@dump-sem-ir-end

// --- fail_extra_text.carbon
// CHECK:STDOUT: - filename: fail_extra_text.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_extra_text.carbon:[[@LINE+4]]:3: error: whitespace is required after '//' [NoWhitespaceAfterCommentIntroducer]
// CHECK:STDERR: //@dump-sem-ir-begin more text
// CHECK:STDERR:   ^
// CHECK:STDERR:
//@dump-sem-ir-begin more text

// --- fail_start_only.carbon
// CHECK:STDOUT: - filename: fail_start_only.carbon
// CHECK:STDOUT:   tokens:
// CHECK:STDOUT:   dump_sem_ir_ranges:
// CHECK:STDOUT:   - {begin: 1, end: 0}

//@dump-sem-ir-begin
// CHECK:STDERR: fail_start_only.carbon:[[@LINE+4]]:1: error: missing `//@dump-sem-ir-end` to match `//@dump-sem-ir-begin` [DumpSemIRRangeMissingEnd]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_end_only.carbon
// CHECK:STDOUT: - filename: fail_end_only.carbon
// CHECK:STDOUT:   tokens:

// CHECK:STDERR: fail_end_only.carbon:[[@LINE+4]]:1: error: missing `//@dump-sem-ir-begin` to match `//@dump-sem-ir-end` [DumpSemIRRangeMissingBegin]
// CHECK:STDERR: //@dump-sem-ir-end
// CHECK:STDERR: ^
// CHECK:STDERR:
//@dump-sem-ir-end

// --- fail_misordered.carbon
// CHECK:STDOUT: - filename: fail_misordered.carbon
// CHECK:STDOUT:   tokens:

a
// CHECK:STDOUT:   - { index: 1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "a", identifier: 0, has_leading_space: true }
//@dump-sem-ir-begin
b
// CHECK:STDOUT:   - { index: 2, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "b", identifier: 1, has_leading_space: true }
// CHECK:STDERR: fail_misordered.carbon:[[@LINE+4]]:1: error: missing `//@dump-sem-ir-end` to match `//@dump-sem-ir-begin` [DumpSemIRRangeMissingEnd]
// CHECK:STDERR: //@dump-sem-ir-begin
// CHECK:STDERR: ^
// CHECK:STDERR:
//@dump-sem-ir-begin
c
// CHECK:STDOUT:   - { index: 3, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "c", identifier: 2, has_leading_space: true }
//@dump-sem-ir-end
d
// CHECK:STDOUT:   - { index: 4, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "d", identifier: 3, has_leading_space: true }
// CHECK:STDERR: fail_misordered.carbon:[[@LINE+4]]:1: error: missing `//@dump-sem-ir-begin` to match `//@dump-sem-ir-end` [DumpSemIRRangeMissingBegin]
// CHECK:STDERR: //@dump-sem-ir-end
// CHECK:STDERR: ^
// CHECK:STDERR:
//@dump-sem-ir-end
e
// CHECK:STDOUT:   - { index: 5, kind: "Identifier", line: {{ *}}[[@LINE-1]], column: 1, indent: 1, spelling: "e", identifier: 4, has_leading_space: true }
// CHECK:STDOUT:   dump_sem_ir_ranges:
// CHECK:STDOUT:   - {begin: 2, end: 2}
// CHECK:STDOUT:   - {begin: 3, end: 3}

// --- fail_count_mismatch.carbon
// CHECK:STDOUT: - filename: fail_count_mismatch.carbon
// CHECK:STDOUT:   tokens:

a
// CHECK:STDOUT:   - { index: 1, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "a", identifier: 0, has_leading_space: true }
//@dump-sem-ir-begin
b
// CHECK:STDOUT:   - { index: 2, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "b", identifier: 1, has_leading_space: true }
//@dump-sem-ir-end
c
// CHECK:STDOUT:   - { index: 3, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "c", identifier: 2, has_leading_space: true }
//@dump-sem-ir-begin
d
// CHECK:STDOUT:   - { index: 4, kind: "Identifier", line: {{ *}}[[@LINE-1]], column:  1, indent: 1, spelling: "d", identifier: 3, has_leading_space: true }
// CHECK:STDOUT:   dump_sem_ir_ranges:
// CHECK:STDOUT:   - {begin: 2, end: 2}
// CHECK:STDOUT:   - {begin: 4, end: 4}

// CHECK:STDERR: fail_count_mismatch.carbon:[[@LINE+3]]:17: error: missing `//@dump-sem-ir-end` to match `//@dump-sem-ir-begin` [DumpSemIRRangeMissingEnd]
// CHECK:STDERR: // CHECK:STDERR:
// CHECK:STDERR:                 ^
// CHECK:STDERR:
