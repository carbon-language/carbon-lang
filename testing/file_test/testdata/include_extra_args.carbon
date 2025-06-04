// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: foo
// INCLUDE-FILE: testing/file_test/testdata/include_files/extra_args.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //testing/file_test:file_test_base_test --test_arg=--file_tests=testing/file_test/testdata/include_extra_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //testing/file_test:file_test_base_test -- --dump_output --file_tests=testing/file_test/testdata/include_extra_args.carbon

// CHECK:STDOUT: 6 args: `default_args`, `include_extra_args.carbon`, `include_files/extra_args.carbon`, `foo`, `bar`, `baz`
