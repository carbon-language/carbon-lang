// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/fail_modifiers.carbon

virtual namespace B

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `namespace` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: impl namespace
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl namespace

// CHECK:STDOUT: - filename: fail_modifiers.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─VirtualModifier 'virtual'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   ├─Namespace 'namespace' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
