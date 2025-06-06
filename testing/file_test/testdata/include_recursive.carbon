// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: testing/file_test/testdata/include_files/recursive.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/include_recursive.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/include_recursive.carbon

// CHECK:STDOUT: 6 args: `default_args`, `include_recursive.carbon`, `c.carbon`, `d.carbon`, `a.carbon`, `b.carbon`
// CHECK:STDOUT: c.carbon:2: c
// CHECK:STDOUT: d.carbon:2: d
// CHECK:STDOUT: a.carbon:2: a
// CHECK:STDOUT: b.carbon:2: b
