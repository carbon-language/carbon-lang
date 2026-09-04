// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_definition_in_impl_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_definition_in_impl_file.carbon

// --- decl_in_api_definition_in_impl.carbon

library "[[@TEST_NAME]]";

fn A();

// --- decl_in_api_definition_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

fn A() {}

// --- use_decl_in_api.carbon

library "[[@TEST_NAME]]";

// --- use_decl_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

import library "decl_in_api_definition_in_impl";

// --- decl_only_in_api.carbon

library "[[@TEST_NAME]]";

fn B();

// --- decl_only_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

// --- decl_in_api_decl_in_impl.carbon

library "[[@TEST_NAME]]";

fn C();

// --- fail_decl_in_api_decl_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE+12]]:1: error: redeclaration of `fn C` is redundant [RedeclRedundant]
// CHECK:STDERR: fn C();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: decl_in_api_decl_in_impl.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn C();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE+4]]:1: error: no definition found for declaration in impl file [MissingDefinitionInImpl]
// CHECK:STDERR: fn C();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn C();

// --- decl_in_api_decl_and_definition_in_impl.carbon

library "[[@TEST_NAME]]";

fn D();

// --- fail_decl_in_api_decl_and_definition_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_in_api_decl_and_definition_in_impl.impl.carbon:[[@LINE+8]]:1: error: redeclaration of `fn D` is redundant [RedeclRedundant]
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_decl_in_api_decl_and_definition_in_impl.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: decl_in_api_decl_and_definition_in_impl.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn D();

fn D() {}

// --- decl_only_in_impl.carbon

library "[[@TEST_NAME]]";

// --- fail_decl_only_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_only_in_impl.impl.carbon:[[@LINE+4]]:1: error: no definition found for declaration in impl file [MissingDefinitionInImpl]
// CHECK:STDERR: fn E();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn E();
