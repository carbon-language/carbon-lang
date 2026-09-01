// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_definition_in_impl_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_definition_in_impl_file.carbon

// --- decl_in_api_definition_in_impl.carbon

library "[[@TEST_NAME]]";

interface A;

// --- fail_decl_in_api_definition_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_in_api_definition_in_impl.impl.carbon:[[@LINE+8]]:1: error: redeclaration of `interface A` is redundant [RedeclRedundant]
// CHECK:STDERR: interface A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_decl_in_api_definition_in_impl.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: decl_in_api_definition_in_impl.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: interface A;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
interface A;

interface A {}

// --- use_decl_in_api.carbon

library "[[@TEST_NAME]]";

// --- use_decl_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

import library "decl_in_api_definition_in_impl";

// --- decl_only_in_api.carbon

library "[[@TEST_NAME]]";

interface B;

// --- decl_only_in_api.impl.carbon

impl library "[[@TEST_NAME]]";

// --- decl_in_api_decl_in_impl.carbon

library "[[@TEST_NAME]]";

interface C;

// --- fail_decl_in_api_decl_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE+12]]:1: error: redeclaration of `interface C` is redundant [RedeclRedundant]
// CHECK:STDERR: interface C;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: decl_in_api_decl_in_impl.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: interface C;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_decl_in_api_decl_in_impl.impl.carbon:[[@LINE+4]]:1: error: no definition found for declaration in impl file [MissingDefinitionInImpl]
// CHECK:STDERR: interface C;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
interface C;

// --- decl_only_in_impl.carbon

library "[[@TEST_NAME]]";

// --- fail_decl_only_in_impl.impl.carbon

impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_decl_only_in_impl.impl.carbon:[[@LINE+4]]:1: error: no definition found for declaration in impl file [MissingDefinitionInImpl]
// CHECK:STDERR: interface D;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
interface D;
