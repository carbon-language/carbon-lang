// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_empty.carbon

// CHECK:STDERR: fail_empty.carbon:[[@LINE+2]]:4: error: expected pattern [ExpectedPattern]
// CHECK:STDERR:
let;

// CHECK:STDOUT: - filename: fail_empty.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
