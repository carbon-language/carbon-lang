// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/three_stars.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/three_stars.carbon

fn F() {
  const T* * *p;
}

// CHECK:STDOUT: - filename: three_stars.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │     ╭─PrefixOperatorConst 'const'
// CHECK:STDOUT:   │ │   ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │   │ ╭─IdentifierNameExpr 'p'
// CHECK:STDOUT:   │ │   ├─PrefixOperatorStar '*'
// CHECK:STDOUT:   │ │ ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
