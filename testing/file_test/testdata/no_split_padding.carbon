// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SET-NO-SPLIT-PADDING
//
// Disable autoupdate because we want the raw line numbers.
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/no_split_padding.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/no_split_padding.carbon
// CHECK:STDOUT: 3 args: `default_args`, `a.carbon`, `b.carbon`

// --- a.carbon
// CHECK:STDOUT: a.carbon:2: aaa
aaa


// --- b.carbon
// CHECK:STDOUT: b.carbon:2: bbb
bbb
