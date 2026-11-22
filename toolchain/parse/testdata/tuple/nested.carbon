// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/tuple/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/tuple/nested.carbon

var y: ((), (), ());

var (a: (), (b: (), c: ()));

// CHECK:STDOUT: - filename: nested.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │   │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │   │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameNotBeforeSignature 'c'
// CHECK:STDOUT:   │ │   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │   │ ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │   ├─TuplePattern ')'
// CHECK:STDOUT:   │ │ ╭─TuplePattern ')'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
