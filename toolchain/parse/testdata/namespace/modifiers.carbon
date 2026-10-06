// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/namespace/modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/namespace/modifiers.carbon

private extern abstract base default final namespace Foo;

impl namespace Foo;

// CHECK:STDOUT: - filename: modifiers.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─PrivateModifier 'private'
// CHECK:STDOUT:   │ ├─ExternModifier 'extern'
// CHECK:STDOUT:   │ ├─AbstractModifier 'abstract'
// CHECK:STDOUT:   │ ├─BaseModifier 'base'
// CHECK:STDOUT:   │ ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ ├─FinalModifier 'final'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─Namespace ';'
// CHECK:STDOUT:   │ ╭─NamespaceStart 'namespace'
// CHECK:STDOUT:   │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─Namespace ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
