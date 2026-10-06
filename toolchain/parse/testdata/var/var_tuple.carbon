// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/var_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/var_tuple.carbon

fn F() {
  var (s: str, i: 32) = ("hello", 0);
}

// CHECK:STDOUT: - filename: var_tuple.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameNotBeforeSignature 'i'
// CHECK:STDOUT:   │ │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ├─IntLiteral '32'
// CHECK:STDOUT:   │ │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ╭─TuplePattern ')'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─StringLiteral '"hello"'
// CHECK:STDOUT:   │ │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '0'
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
