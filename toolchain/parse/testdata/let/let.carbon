// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/let.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/let.carbon

let v: i32 = 0;
let _: i32 = 1;
fn F() {
  let s: str = "hello";
  let _: str = "goodbye";
}

// CHECK:STDOUT: - filename: let.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameNotBeforeSignature 'v'
// CHECK:STDOUT:   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '0'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ ╭─UnderscoreName '_'
// CHECK:STDOUT:   │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─LetInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '1'
// CHECK:STDOUT:   ├─LetDecl ';'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 's'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ├─StringLiteral '"hello"'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   │ │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─UnderscoreName '_'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─LetInitializer '='
// CHECK:STDOUT:   │ │ ├─StringLiteral '"goodbye"'
// CHECK:STDOUT:   │ ├─LetDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
