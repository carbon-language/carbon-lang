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

// `impl` declarations only match when they are the same syntactically.

// --- fail_match_with_associated_type.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }

// CHECK:STDERR: fail_match_with_associated_type.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as I where .T = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I where .T = {};
impl () as I where .T = {} and .T impls (type where .Self impls type) {}

// CHECK:STDERR: fail_match_with_associated_type.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl {} as I;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl {} as I;
impl {} as I where .T = {} and .T impls (type where .Self impls type) {}

// CHECK:STDERR: fail_match_with_associated_type.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl ({},) as I where .T = {} and .T impls (type where .Self impls type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl ({},) as I where .T = {} and .T impls (type where .Self impls type);
impl ({},) as I where .T = {} {}

// --- fail_match_with_empty_interface.carbon
library "[[@TEST_NAME]]";

interface J {}

// CHECK:STDERR: fail_match_with_empty_interface.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as J;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as J;
impl () as J where .Self impls type and .Self impls (type where .Self impls type) {}

// CHECK:STDERR: fail_match_with_empty_interface.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl {} as J where .Self impls type and .Self impls (type where .Self impls type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl {} as J where .Self impls type and .Self impls (type where .Self impls type);
impl {} as J {}

// --- fail_parens_other_nesting.carbon
library "[[@TEST_NAME]]";

interface K {}

// CHECK:STDERR: fail_parens_other_nesting.carbon:[[@LINE+4]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl {} as (K where .Self impls type) where .Self impls type;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
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
