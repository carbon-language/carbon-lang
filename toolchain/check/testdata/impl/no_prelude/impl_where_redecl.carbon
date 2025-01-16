// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_where_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_where_redecl.carbon

// --- match.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }
interface J {}

// `impl` matching ignores the `where` clause. It should not get confused by
// nested `where`, so the following should not trigger impl declaration without
// definition diagnostics.

impl () as I where .T = {};
impl () as I where .T = {} and .T impls (type where .Self impls type) {}

impl {} as I;
impl {} as I where .T = {} and .T impls (type where .Self impls type) {}

impl ({},) as I where .T = {} and .T impls (type where .Self impls type);
impl ({},) as I where .T = {} {}

impl () as J;
impl () as J where .Self impls type and .Self impls (type where .Self impls type) {}

impl {} as J where .Self impls type and .Self impls (type where .Self impls type);
impl {} as J {}

// --- parens_other_nesting.carbon
library "[[@TEST_NAME]]";

interface K {}

// `impl` matching only ignores a root-level `where` clause.

impl {} as (K where .Self impls type) where .Self impls type;
impl {} as (K where .Self impls type) {}

// --- fail_other_nesting.carbon
library "[[@TEST_NAME]]";

interface L {}

// CHECK:STDERR: fail_other_nesting.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as (L where .Self impls type) where .Self impls type;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as (L where .Self impls type) where .Self impls type;
impl () as L {}
