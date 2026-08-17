// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/associated_constants.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/associated_constants.carbon

// --- associated_constants.carbon

library "[[@TEST_NAME]]";

interface I {
  let T: type;
  final let I: i32;
  default let D: bool;
}

// --- initializer.carbon

library "[[@TEST_NAME]]";

interface I {
  let T: type = (i32, i32);
  final let I: i32 = 42;
  default let D: bool = false;
}

// CHECK:STDOUT: - filename: associated_constants.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"associated_constants"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─FinalModifier 'final'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'D'
// CHECK:STDOUT:   │ │ │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: initializer.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LibraryIntroducer 'library'
// CHECK:STDOUT:   │ ├─LibraryName '"initializer"'
// CHECK:STDOUT:   ├─LibraryDecl ';'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantInitializer '='
// CHECK:STDOUT:   │ │ │ ╭─TupleLiteralStart '('
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ │ ├─TupleLiteralComma ','
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─TupleLiteral ')'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─FinalModifier 'final'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantInitializer '='
// CHECK:STDOUT:   │ │ ├─IntLiteral '42'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ │ ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'D'
// CHECK:STDOUT:   │ │ │ ├─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantNameAndType ':'
// CHECK:STDOUT:   │ │ ├─AssociatedConstantInitializer '='
// CHECK:STDOUT:   │ │ ├─BoolLiteralFalse 'false'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
