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

// --- fail_where_nested_inside_where_through_alias_impls_lhs.carbon
library "[[@TEST_NAME]]";

interface I {}
class C(T:! type);

alias A = I where .Self == C;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_lhs.carbon:[[@LINE+7]]:21: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where C(A) impls type) {}
// CHECK:STDERR:                     ^~~~
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_lhs.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where C(A) impls type) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where C(A) impls type) {}

// --- fail_where_nested_inside_where_through_alias_impls_rhs_with_self_impls_interface.carbon
library "[[@TEST_NAME]]";

interface I {}

alias A = type where .Self impls I;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_self_impls_interface.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_self_impls_interface.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls A) {}

// --- fail_where_nested_inside_where_through_alias_impls_rhs_with_self_impls_named_constraint.carbon
library "[[@TEST_NAME]]";

constraint N {}

alias A = type where .Self impls N;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_self_impls_named_constraint.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_self_impls_named_constraint.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls A) {}

// --- fail_where_nested_inside_where_through_alias_impls_rhs_with_type_impls_interface.carbon
library "[[@TEST_NAME]]";

interface I(T:! type) {}
class C;

alias A = type where C impls I(.Self);

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_type_impls_interface.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_type_impls_interface.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls A) {}

// --- fail_where_nested_inside_where_through_alias_impls_rhs_with_type_impls_named_constraint.carbon
library "[[@TEST_NAME]]";

constraint N(T:! type) {}
class C;

alias A = type where C impls N(.Self);

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_type_impls_named_constraint.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_type_impls_named_constraint.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls A) {}

// --- fail_where_nested_inside_where_through_alias_impls_rhs_with_rewrite.carbon
library "[[@TEST_NAME]]";

interface I { let I1:! type; }
class C;

alias A = I where .I1 = C;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_rewrite.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_rewrite.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls A) {}

// --- fail_where_nested_inside_where_through_alias_impls_rhs_with_equiv.carbon
library "[[@TEST_NAME]]";

class C;

alias A = type where .Self == C;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_equiv.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_impls_rhs_with_equiv.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls A) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls A) {}

// --- fail_where_nested_inside_where_through_alias_rewrite_rhs.carbon
library "[[@TEST_NAME]]";

interface I { let I1:! type; }
class C;

alias A = I where .Self == C;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_rewrite_rhs.carbon:[[@LINE+7]]:24: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! I where .I1 = A) {}
// CHECK:STDERR:                        ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_rewrite_rhs.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! I where .I1 = A) {}
// CHECK:STDERR:          ^~~~~~~
// CHECK:STDERR:
fn F(_:! I where .I1 = A) {}

// --- fail_where_nested_inside_where_through_alias_equiv_lhs.carbon
library "[[@TEST_NAME]]";

interface I {}
class C;

alias A = I where .Self == C;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_equiv_lhs.carbon:[[@LINE+7]]:21: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where A == .Self) {}
// CHECK:STDERR:                     ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_equiv_lhs.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where A == .Self) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where A == .Self) {}

// --- fail_where_nested_inside_where_through_alias_equiv_rhs.carbon
library "[[@TEST_NAME]]";

interface I {}
class C;

alias A = I where .Self == C;

// The use of `A` introduces a `where` expression inside the `where` written
// here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_equiv_rhs.carbon:[[@LINE+7]]:30: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self == A) {}
// CHECK:STDERR:                              ^
// CHECK:STDERR: fail_where_nested_inside_where_through_alias_equiv_rhs.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self == A) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self == A) {}

// --- fail_where_nested_inside_where_through_fn_eval.carbon
library "[[@TEST_NAME]]";

interface I {}
class C;

eval fn E() -> type {
    return I where .Self == C;
}

// The use of `E()` introduces a `where` expression inside the `where`
// written here, which is an error.
//
// CHECK:STDERR: fail_where_nested_inside_where_through_fn_eval.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls E()) {}
// CHECK:STDERR:                                 ^~~
// CHECK:STDERR: fail_where_nested_inside_where_through_fn_eval.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls E()) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls E()) {}

// --- fail_where_nested_inside_where_through_operator_eval.carbon
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
// CHECK:STDERR: fail_where_nested_inside_where_through_operator_eval.carbon:[[@LINE+7]]:33: error: found `where` expression nested on the right-hand side of `where` [NestedWhereInsideWhere]
// CHECK:STDERR: fn F(_:! type where .Self impls type * X) {}
// CHECK:STDERR:                                 ^~~~~~~~
// CHECK:STDERR: fail_where_nested_inside_where_through_operator_eval.carbon:[[@LINE+4]]:10: note: on right-hand side of `where` here [NestedWhereInsideWhereOuterNote]
// CHECK:STDERR: fn F(_:! type where .Self impls type * X) {}
// CHECK:STDERR:          ^~~~~~~~~~
// CHECK:STDERR:
fn F(_:! type where .Self impls type * X) {}
