// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/fail_modifiers.carbon

// --- fail_virtual_class.carbon

virtual class B
// CHECK:STDERR: fail_virtual_class.carbon:[[@LINE+4]]:1: error: `virtual` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_impl_class.carbon

// CHECK:STDERR: fail_impl_class.carbon:[[@LINE+4]]:1: error: `impl` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: impl class
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl class

// --- fail_extend_adapt_base.carbon

class C {
  // CHECK:STDERR: fail_extend_adapt_base.carbon:[[@LINE+4]]:16: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   extend adapt base B;
  // CHECK:STDERR:                ^~~~
  // CHECK:STDERR:
  extend adapt base B;
}

// CHECK:STDOUT: - filename: fail_virtual_class.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ ├─VirtualModifier 'virtual'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   ├─ClassDecl 'B' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_impl_class.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │ ├─ImplModifier 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse '' has_error
// CHECK:STDOUT:   ├─ClassDecl 'class' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_extend_adapt_base.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AdaptIntroducer 'adapt'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ ├─InvalidParse 'base' has_error
// CHECK:STDOUT:   │ ├─AdaptDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
