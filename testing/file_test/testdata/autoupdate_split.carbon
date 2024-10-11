// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/autoupdate_split.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/autoupdate_split.carbon

// --- a.carbon
aaa

// --- b.carbon
bbb

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: 3 args: `default_args`, `a.carbon`, `b.carbon`
// CHECK:STDOUT: a.carbon:1: aaa
// CHECK:STDOUT: b.carbon:1: bbb
