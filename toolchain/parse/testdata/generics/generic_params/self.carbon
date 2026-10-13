// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/generic_params/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/generic_params/self.carbon

// --- generic_self.carbon

class C {
  fn A(generic self: Self);
  fn B(template self: Self);
}

// --- fail_todo_generic_self_no_type.carbon

class C {
  // CHECK:STDERR: fail_todo_generic_self_no_type.carbon:[[@LINE+4]]:20: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
  // CHECK:STDERR:   fn A(generic self);
  // CHECK:STDERR:                    ^
  // CHECK:STDERR:
  fn A(generic self);
  // CHECK:STDERR: fail_todo_generic_self_no_type.carbon:[[@LINE+4]]:21: error: expected `:` or `:?` in binding pattern [ExpectedBindingPattern]
  // CHECK:STDERR:   fn B(template self);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  fn B(template self);
}

// CHECK:STDOUT: - filename: generic_self.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'A'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'B'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │   ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ╭─TemplateBindingName 'template'
// CHECK:STDOUT:   │ │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ │ ├─SelfTypeNameExpr 'Self'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_todo_generic_self_no_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'A'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ │ │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern 'generic' has_error
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   │ │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ │ ├─IdentifierNameMaybeBeforeSignature 'B'
// CHECK:STDOUT:   │ │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ │ ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │ │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ │ │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │ │ ├─LetBindingPattern 'template' has_error
// CHECK:STDOUT:   │ │ ├─ExplicitParamList ')' has_error
// CHECK:STDOUT:   │ ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
