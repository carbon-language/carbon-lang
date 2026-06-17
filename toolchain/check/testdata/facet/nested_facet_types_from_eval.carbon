// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/nested_facet_types_from_eval.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/nested_facet_types_from_eval.carbon

// --- todo_fail_where_nested_inside_where_through_alias.carbon
library "[[@TEST_NAME]]";

interface I {}
class C;

alias A = I where .Self == C;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// TODO This should fail.
fn F(_:! type where .Self impls A) {}

// --- todo_fail_where_nested_inside_where_through_fn_eval.carbon
library "[[@TEST_NAME]]";

interface I {}
class C;

eval fn E() -> type {
    return I where .Self == C;
}

// The use of `E()` introduces a `where` expression inside the `where`
// written here, which is an error.
//
// TODO This should fail.
fn F(_:! type where .Self impls E()) {}

// --- todo_fail_where_nested_inside_where_through_operator_eval.carbon
library "[[@TEST_NAME]]";

interface I {}
class C;

class X1 {
    impl type as Core.MulWith(X1) where .Result = type {
        eval fn Op(unused self, _: X1) -> type {
            return I where .Self == C;
        }
    }
}
alias X = {} as X1;

// The use of `type * X` introduces a `where` expression inside the `where`
// written here, which is an error.
//
// TODO This should fail.
fn F(_:! type where .Self impls type * X) {}
