// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/nested.carbon

namespace Foo;
namespace Foo.Bar;

fn Foo.Bar.Baz() {
  Foo.Bar.Wiz();
}

// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─Namespace ';'
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Bar'
// CHECK:STDOUT:   ├─Namespace ';'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │   ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameNotBeforeSignature 'Bar'
// CHECK:STDOUT:   │   ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'Baz'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │         ╭─IdentifierNameExpr 'Foo'
// CHECK:STDOUT:   │ │         ├─IdentifierNameNotBeforeSignature 'Bar'
// CHECK:STDOUT:   │ │       ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │       ├─IdentifierNameNotBeforeSignature 'Wiz'
// CHECK:STDOUT:   │ │     ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
