// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/named_constraint/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/named_constraint/basic.carbon

// --- empty.carbon

constraint Empty {}

// --- with_require.carbon

constraint WithRequire {
  // TODO: Add a require statement.
}

// --- forward_decl.carbon

constraint ForwardDeclared;

// CHECK:STDOUT: - filename: empty.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Empty'
// CHECK:STDOUT:   │ ╭─NamedConstraintDefinitionStart '{'
// CHECK:STDOUT:   ├─NamedConstraintDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: with_require.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'WithRequire'
// CHECK:STDOUT:   │ ╭─NamedConstraintDefinitionStart '{'
// CHECK:STDOUT:   ├─NamedConstraintDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: forward_decl.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─NamedConstraintIntroducer 'constraint'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'ForwardDeclared'
// CHECK:STDOUT:   ├─NamedConstraintDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
