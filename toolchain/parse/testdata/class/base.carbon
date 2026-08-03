// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/base.carbon

base class B {}

class D {
  base: B1;
  extend base: B2;
  base class Nested;
  private base class PrivateNested;
}

// CHECK:STDOUT: - filename: base.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─BaseModifier 'base'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'D'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ │ ├─BaseColon ':'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'B1'
// CHECK:STDOUT:   │ ├─BaseDecl ';'
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ ├─BaseColon ':'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'B2'
// CHECK:STDOUT:   │ ├─BaseDecl ';'
// CHECK:STDOUT:   │ │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ │ ├─BaseModifier 'base'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'Nested'
// CHECK:STDOUT:   │ ├─ClassDecl ';'
// CHECK:STDOUT:   │ │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ │ ├─PrivateModifier 'private'
// CHECK:STDOUT:   │ │ ├─BaseModifier 'base'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'PrivateNested'
// CHECK:STDOUT:   │ ├─ClassDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
