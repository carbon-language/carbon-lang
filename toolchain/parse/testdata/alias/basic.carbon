// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/alias/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/alias/basic.carbon

alias a = foo;
alias b = 0;
alias c = 1 + 3;

alias NS.ns = foo;

class C {
  alias c = foo;
}

fn F() {
  alias f = foo;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   ├─Alias ';'
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '0'
// CHECK:STDOUT:   ├─Alias ';'
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ │ ╭─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─IntLiteral '3'
// CHECK:STDOUT:   │ ├─InfixOperatorPlus '+'
// CHECK:STDOUT:   ├─Alias ';'
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'NS'
// CHECK:STDOUT:   │ ├─IdentifierNameQualifierWithoutParams '.'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'ns'
// CHECK:STDOUT:   │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   ├─Alias ';'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ ├─Alias ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'f'
// CHECK:STDOUT:   │ │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   │ ├─Alias ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
