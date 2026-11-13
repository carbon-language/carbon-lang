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

// CHECK:STDERR:        ╭─┤ fail_forward_decl.carbon:[[@LINE+7]]:8
// CHECK:STDERR: ->     │ fn F() -> auto;
// CHECK:STDERR:        ·        ━━━┳━━━
// CHECK:STDERR:        ·           ┗━━┫ error: forward declaration of function with deduced return type
// CHECK:STDERR:        ·              ┃        [DeducedReturnTypeNotDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> auto;

// --- fail_forward_decl_then_def.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_forward_decl_then_def.carbon:[[@LINE+7]]:8
// CHECK:STDERR: ->     │ fn F() -> auto;
// CHECK:STDERR:        ·        ━━━┳━━━
// CHECK:STDERR:        ·           ┗━━┫ error: forward declaration of function with deduced return type
// CHECK:STDERR:        ·              ┃        [DeducedReturnTypeNotDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> auto;
// CHECK:STDERR:        ╭─┤ fail_forward_decl_then_def.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │ fn F() -> auto;
// CHECK:STDERR:        · ───────┬───────
// CHECK:STDERR:        ·        ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ━━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: redeclaration of function with deduced return type
// CHECK:STDERR:        ·            ┃        [DeducedReturnTypeRedecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }

// --- fail_decl_then_auto_def.carbon
library "[[@TEST_NAME]]";

fn F() -> i32;
// CHECK:STDERR:        ╭─┤ fail_decl_then_auto_def.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │ fn F() -> i32;
// CHECK:STDERR:        · ───────┬──────
// CHECK:STDERR:        ·        ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ━━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: redeclaration of function with deduced return type
// CHECK:STDERR:        ·            ┃        [DeducedReturnTypeRedecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }

// --- fail_auto_def_then_decl.carbon
library "[[@TEST_NAME]]";

fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        ╭─┤ fail_auto_def_then_decl.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │ fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ────────┬───────
// CHECK:STDERR:        ·         ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn F() -> i32;
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: redeclaration of function with deduced return type
// CHECK:STDERR:        ·           ┃        [DeducedReturnTypeRedecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> i32;

// --- fail_redefinition.carbon
library "[[@TEST_NAME]]";

fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        ╭─┤ fail_redefinition.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │ fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ────────┬───────
// CHECK:STDERR:        ·         ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ━━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: redeclaration of function with deduced return type
// CHECK:STDERR:        ·            ┃        [DeducedReturnTypeRedecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }

// --- fail_member_out_of_line.carbon
library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR:        ╭─┤ fail_member_out_of_line.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   fn F() -> auto;
  // CHECK:STDERR:        ·          ━━━┳━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: forward declaration of function with deduced return type
  // CHECK:STDERR:        ·                ┃        [DeducedReturnTypeNotDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  fn F() -> auto;
}

// CHECK:STDERR:        ╭─┤ fail_member_out_of_line.carbon:[[@LINE+11]]:1
// CHECK:STDERR:        │   fn F() -> auto;
// CHECK:STDERR:        ·   ───────┬───────
// CHECK:STDERR:        ·          ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn C.F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ━━━━━━━━━┳━━━━━━━━
// CHECK:STDERR:        ·          ┗━━┫ error: redeclaration of function with deduced return type
// CHECK:STDERR:        ·             ┃        [DeducedReturnTypeRedecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn C.F() -> auto { return 0 as i32; }

// --- fail_builtin.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_builtin.carbon:[[@LINE+14]]:26
// CHECK:STDERR: ->     │ fn F(n: Core.IntLiteral) -> auto = "int.make_type_signed";
// CHECK:STDERR:        ·                          ━━┳━━━━
// CHECK:STDERR:        ·                            ┗━━┫ error: builtin function with deduced return type
// CHECK:STDERR:        ·                               ┃        [DeducedReturnTypeNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_builtin.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn F(n: Core.IntLiteral) -> auto = "int.make_type_signed";
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: invalid signature for builtin function "int.make_type_signed"
// CHECK:STDERR:        ·           ┃        [InvalidBuiltinSignature]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(n: Core.IntLiteral) -> auto = "int.make_type_signed";

// --- fail_extern.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_extern.carbon:[[@LINE+7]]:15
// CHECK:STDERR: ->     │ extern fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        ·               ━━━┳━━━
// CHECK:STDERR:        ·                  ┗━━┫ error: `extern` function with deduced return type
// CHECK:STDERR:        ·                     ┃        [DeducedReturnTypeNotAllowed]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern fn F() -> auto { return 0 as i32; }

// --- api_def.carbon
library "[[@TEST_NAME]]";

fn F() -> auto { return 0 as i32; }

// --- fail_api_def.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_api_def.impl.carbon:[[@LINE+19]]:1
// CHECK:STDERR: ->     │ fn F() -> i32;
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: redeclaration of function with deduced return type
// CHECK:STDERR:        ·           ┃        [DeducedReturnTypeRedecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ api_def.carbon
// CHECK:STDERR:        │ fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ────────┬───────
// CHECK:STDERR:        ·         ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_api_def.impl.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn F() -> i32;
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: no definition found for declaration in impl file
// CHECK:STDERR:        ·           ┃        [MissingDefinitionInImpl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> i32;

// --- api_decl.carbon
library "[[@TEST_NAME]]";

fn F() -> i32;

// --- fail_api_decl.impl.carbon
impl library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_api_decl.impl.carbon:[[@LINE+12]]:1
// CHECK:STDERR: ->     │ fn F() -> auto { return 0 as i32; }
// CHECK:STDERR:        · ━━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: redeclaration of function with deduced return type
// CHECK:STDERR:        ·            ┃        [DeducedReturnTypeRedecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ api_decl.carbon
// CHECK:STDERR:        │ fn F() -> i32;
// CHECK:STDERR:        · ───────┬──────
// CHECK:STDERR:        ·        ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> auto { return 0 as i32; }
