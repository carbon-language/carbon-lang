// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/file_only_re_one_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/file_only_re_one_file.carbon
// CHECK:STDOUT: 2 args: `default_args`, `file_only_re_one_file.carbon`
// CHECK:STDOUT: unattached message 1
// CHECK:STDOUT: file: file_only_re_one_file.carbon
// CHECK:STDOUT: line: [[@LINE-12]]
// CHECK:STDOUT: unattached message 2
