// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/base_misplaced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/base_misplaced.carbon

base class B {}

// These are rejected by check.
base: B;

fn F() {
  base: B;
}


// CHECK:STDOUT: - filename: base_misplaced.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─BaseModifier 'base'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ ├─BaseColon ':'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   ├─BaseDecl ';'
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ │ ├─BaseColon ':'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ ├─BaseDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
