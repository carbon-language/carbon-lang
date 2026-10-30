// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/package_expr/fail_in_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/package_expr/fail_in_name.carbon

// CHECK:STDERR:        ╭─┤ fail_in_name.carbon:[[@LINE+7]]:14
// CHECK:STDERR: ->     │ var package.n: i32;
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: unexpected expression before : in binding pattern
// CHECK:STDERR:        ·                 ┃        [ExpectedBindingName]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var package.n: i32;

// `val` is a keyword, but during error recovery we treat it as an identifier.
// CHECK:STDERR:        ╭─┤ fail_in_name.carbon:[[@LINE+15]]:13
// CHECK:STDERR: ->     │ var package.val: i32;
// CHECK:STDERR:        ·             ┳━━
// CHECK:STDERR:        ·             ┗━━┫ error: expected identifier after `.`; prefix reserved word with
// CHECK:STDERR:        ·                ┃        `r#` to form a valid identifier
// CHECK:STDERR:        ·                ┃        [ExpectedIdentifierAfterPeriodOrArrow]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_in_name.carbon:[[@LINE+7]]:16
// CHECK:STDERR: ->     │ var package.val: i32;
// CHECK:STDERR:        ·                ┳
// CHECK:STDERR:        ·                ┗━━┫ error: unexpected expression before : in binding pattern
// CHECK:STDERR:        ·                   ┃        [ExpectedBindingName]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var package.val: i32;

// CHECK:STDERR:        ╭─┤ fail_in_name.carbon:[[@LINE+7]]:11
// CHECK:STDERR: ->     │ namespace package.NS;
// CHECK:STDERR:        ·           ━━━┳━━━
// CHECK:STDERR:        ·              ┗━━┫ error: `namespace` introducer should be followed by a name
// CHECK:STDERR:        ·                 ┃        [ExpectedDeclName]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
namespace package.NS;

// CHECK:STDERR:        ╭─┤ fail_in_name.carbon:[[@LINE+7]]:7
// CHECK:STDERR: ->     │ class package.C {
// CHECK:STDERR:        ·       ━━━┳━━━
// CHECK:STDERR:        ·          ┗━━┫ error: `class` introducer should be followed by a name
// CHECK:STDERR:        ·             ┃        [ExpectedDeclName]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class package.C {
}

// CHECK:STDOUT: - filename: fail_in_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─PackageExpr 'package'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─PackageExpr 'package'
// CHECK:STDOUT:   │ │   ├─IdentifierNameNotBeforeSignature 'val'
// CHECK:STDOUT:   │ │ ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─InvalidParse 'package' has_error
// CHECK:STDOUT:   ├─Namespace ';' has_error
// CHECK:STDOUT:   │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ ├─InvalidParse 'package' has_error
// CHECK:STDOUT:   ├─ClassDecl '}' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
