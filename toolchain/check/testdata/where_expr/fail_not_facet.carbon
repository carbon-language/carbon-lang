// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/fail_not_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/fail_not_facet.carbon

// --- fail_left_where_not_facet.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_left_where_not_facet.carbon:[[@LINE+4]]:10: error: left argument of `where` operator must be a facet type [WhereOnNonFacetType]
// CHECK:STDERR: fn F(T:! i32 where .Self == bool);
// CHECK:STDERR:          ^~~
// CHECK:STDERR:
fn F(T:! i32 where .Self == bool);

// --- fail_left_where_unknown.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_left_where_unknown.carbon:[[@LINE+4]]:10: error: name `NOT_DECLARED` not found [NameNotFound]
// CHECK:STDERR: fn G(U:! NOT_DECLARED where .Self == bool);
// CHECK:STDERR:          ^~~~~~~~~~~~
// CHECK:STDERR:
fn G(U:! NOT_DECLARED where .Self == bool);

// --- fail_var.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:8: error: name `e` not found [NameNotFound]
// CHECK:STDERR: var v: e where .x = 3;
// CHECK:STDERR:        ^
// CHECK:STDERR:
var v: e where .x = 3;
