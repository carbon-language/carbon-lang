// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/var_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/var_pattern.carbon

// --- basic.carbon

let var x: () = ();

// --- tuple.carbon

let (x: (), var y: ()) = ((), ());

// --- function.carbon

fn F(x: (), var y: ());

// --- struct.carbon

let {.x = var x: ()} = {.x = ()};

// --- struct_shorthand.carbon

let {var y: ()} = {.y = ()};

// --- fail_nested.carbon

// CHECK:STDERR: fail_nested.carbon:[[@LINE+2]]:25-27: error: `var` nested within another `var` [NestedVar]
// CHECK:STDERR:
let (x: (), var (y: (), var z: ())) = ((), ((), ()));

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: tuple.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─TuplePattern ')'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: function.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ │ ├─StructPatternDesignatedField '='
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: struct_shorthand.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─StructPatternStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─StructPattern '}'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │ ╭─StructFieldDesignator '.'
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─StructLiteralField '='
// CHECK:STDOUT:   │ ├─StructLiteral '}'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_nested.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   ╭─TuplePatternStart '('
// CHECK:STDOUT:   │ │ │   │ ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │ │   │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │   ├─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │   ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │   │   ╭─IdentifierNameNotBeforeSignature 'z'
// CHECK:STDOUT:   │ │ │   │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   │   │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │   │   ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │   │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ │   ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ │ │ ╭─TuplePattern ')' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─TuplePattern ')'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ ├─TupleLiteral ')'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
