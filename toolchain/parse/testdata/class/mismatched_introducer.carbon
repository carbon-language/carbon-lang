// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/mismatched_introducer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/mismatched_introducer.carbon

// Note: invalid modifiers are diagnosed in check not parse.
abstract interface I;
base fn F();
abstract base class C;

// CHECK:STDOUT: - filename: mismatched_introducer.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │ ├─AbstractModifier 'abstract'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   ├─InterfaceDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─BaseModifier 'base'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ ├─AbstractModifier 'abstract'
// CHECK:STDOUT:   │ ├─BaseModifier 'base'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   ├─ClassDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
