// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/fail_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/fail_redecl.carbon

// --- fail_forward_decl.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+4]]:8: error: forward declaration of function with deduced return type [DeducedReturnTypeNotDefinition]
// CHECK:STDERR: fn F() -> auto;
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
fn F() -> auto;

// --- fail_forward_decl_then_def.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_forward_decl_then_def.carbon:[[@LINE+4]]:8: error: forward declaration of function with deduced return type [DeducedReturnTypeNotDefinition]
// CHECK:STDERR: fn F() -> auto;
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
fn F() -> auto;
// CHECK:STDERR: fail_forward_decl_then_def.carbon:[[@LINE+7]]:1: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_forward_decl_then_def.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn F() -> auto;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }

// --- fail_decl_then_auto_def.carbon
library "[[@TEST_NAME]]";

fn F() -> i32;
// CHECK:STDERR: fail_decl_then_auto_def.carbon:[[@LINE+7]]:1: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_then_auto_def.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn F() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }

// --- fail_auto_def_then_decl.carbon
library "[[@TEST_NAME]]";

fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: fail_auto_def_then_decl.carbon:[[@LINE+7]]:1: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fn F() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_auto_def_then_decl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> i32;

// --- fail_redefinition.carbon
library "[[@TEST_NAME]]";

fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+7]]:1: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }

// --- fail_member_out_of_line.carbon
library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_member_out_of_line.carbon:[[@LINE+4]]:10: error: forward declaration of function with deduced return type [DeducedReturnTypeNotDefinition]
  // CHECK:STDERR:   fn F() -> auto;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  fn F() -> auto;
}

// CHECK:STDERR: fail_member_out_of_line.carbon:[[@LINE+7]]:1: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fn C.F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_member_out_of_line.carbon:[[@LINE-6]]:3: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:   fn F() -> auto;
// CHECK:STDERR:   ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn C.F() -> auto { return 0 as i32; }

// --- fail_builtin.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_builtin.carbon:[[@LINE+8]]:26: error: builtin function with deduced return type [DeducedReturnTypeNotAllowed]
// CHECK:STDERR: fn F(n: Core.IntLiteral) -> auto = "int.make_type_signed";
// CHECK:STDERR:                          ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_builtin.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "int.make_type_signed" [InvalidBuiltinSignature]
// CHECK:STDERR: fn F(n: Core.IntLiteral) -> auto = "int.make_type_signed";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(n: Core.IntLiteral) -> auto = "int.make_type_signed";

// --- fail_extern.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_extern.carbon:[[@LINE+4]]:15: error: `extern` function with deduced return type [DeducedReturnTypeNotAllowed]
// CHECK:STDERR: extern fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR:
extern fn F() -> auto { return 0 as i32; }

// --- api_def.carbon
library "[[@TEST_NAME]]";

fn F() -> auto { return 0 as i32; }

// --- fail_api_def.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_api_def.impl.carbon:[[@LINE+12]]:1: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fn F() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_api_def.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: api_def.carbon:3:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_api_def.impl.carbon:[[@LINE+4]]:1: error: no definition found for declaration in impl file [MissingDefinitionInImpl]
// CHECK:STDERR: fn F() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> i32;

// --- api_decl.carbon
library "[[@TEST_NAME]]";

fn F() -> i32;

// --- fail_api_decl.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_api_decl.impl.carbon:[[@LINE+8]]:1: error: redeclaration of function with deduced return type [DeducedReturnTypeRedecl]
// CHECK:STDERR: fn F() -> auto { return 0 as i32; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_api_decl.impl.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: api_decl.carbon:3:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn F() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }
