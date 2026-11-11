// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/fail_not_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/fail_not_facet.carbon

// --- fail_left_where_not_facet.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_left_where_not_facet.carbon:[[@LINE+8]]:17
// CHECK:STDERR: ->     │ fn F(generic T: i32 where .Self == bool);
// CHECK:STDERR:        ·                 ┳━━
// CHECK:STDERR:        ·                 ┗━━┫ error: left argument of `where` operator must be a facet
// CHECK:STDERR:        ·                    ┃        type
// CHECK:STDERR:        ·                    ┃        [WhereOnNonFacetType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(generic T: i32 where .Self == bool);

// --- fail_left_where_unknown.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_left_where_unknown.carbon:[[@LINE+6]]:17
// CHECK:STDERR: ->     │ fn G(generic U: NOT_DECLARED where .Self == bool);
// CHECK:STDERR:        ·                 ━━━━━━┳━━━━━
// CHECK:STDERR:        ·                       ┗━━┫ error: name `NOT_DECLARED` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn G(generic U: NOT_DECLARED where .Self == bool);

// --- fail_var.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_var.carbon:[[@LINE+6]]:8
// CHECK:STDERR: ->     │ var v: e where .x = 3;
// CHECK:STDERR:        ·        ┳
// CHECK:STDERR:        ·        ┗━━┫ error: name `e` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var v: e where .x = 3;
