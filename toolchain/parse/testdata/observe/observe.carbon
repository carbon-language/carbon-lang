// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/observe/observe.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/observe/observe.carbon

// --- base.carbon

observe A == B == C;

// --- impls.carbon

observe T impls I;

// --- equal_equal_impls.carbon

observe T == U impls I;
observe A == B == C == D impls I;

// --- statement.carbon

fn F() {
    observe A == B;
}

interface I {
    observe A == B;
}

// CHECK:STDOUT: - filename: base.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ │ ╭─ObserveEqualEqual '=='
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ ├─ObserveEqualEqual '=='
// CHECK:STDOUT:   ├─ObserveDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ ├─ObserveImpls 'impls'
// CHECK:STDOUT:   ├─ObserveDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: equal_equal_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'U'
// CHECK:STDOUT:   │ │ ╭─ObserveEqualEqual '=='
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ ├─ObserveImpls 'impls'
// CHECK:STDOUT:   ├─ObserveDecl ';'
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ │       ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ │     ╭─ObserveEqualEqual '=='
// CHECK:STDOUT:   │ │     ├─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │   ╭─ObserveEqualEqual '=='
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'D'
// CHECK:STDOUT:   │ │ ╭─ObserveEqualEqual '=='
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ ├─ObserveImpls 'impls'
// CHECK:STDOUT:   ├─ObserveDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: statement.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ │ ├─ObserveEqualEqual '=='
// CHECK:STDOUT:   │ ├─ObserveDecl ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'A'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'B'
// CHECK:STDOUT:   │ │ ├─ObserveEqualEqual '=='
// CHECK:STDOUT:   │ ├─ObserveDecl ';'
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
