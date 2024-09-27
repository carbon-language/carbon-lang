// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/replace_split_content.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/replace_split_content.carbon
// CHECK:STDOUT: 6 args: `default_args`, `a.carbon`, `b.impl.carbon`, `todo_c.carbon`, `todo_fail_d.carbon`, `triplicate.carbon`

// --- a.carbon

library "[[@TEST_NAME]]";
// CHECK:STDOUT: a.carbon:[[@LINE-1]]: library "a";

// --- b.impl.carbon

library "[[@TEST_NAME]]";
// CHECK:STDOUT: b.impl.carbon:[[@LINE-1]]: library "b";

// --- todo_c.carbon

library "[[@TEST_NAME]]";
// CHECK:STDOUT: todo_c.carbon:[[@LINE-1]]: library "c";

// --- todo_fail_d.carbon

library "[[@TEST_NAME]]";
// CHECK:STDOUT: todo_fail_d.carbon:[[@LINE-1]]: library "d";

// --- triplicate.carbon

[[@TEST_NAME]][[@TEST_NAME]][[@TEST_NAME]]
// CHECK:STDOUT: triplicate.carbon:[[@LINE-1]]: triplicatetriplicatetriplicate
