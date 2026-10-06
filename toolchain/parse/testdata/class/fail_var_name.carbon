// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/fail_var_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/fail_var_name.carbon

class C {
  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:6: error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:   var;
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  var;

  var name;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:7: error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:   var :;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  var :;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:7: error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:   var : C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  var : C;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:12: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   var name:;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  var name:;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:12: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   var name C;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  var name C;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:15: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   var name: C C;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  var name: C C;

  var self: C;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:12: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   var name name: C;
  // CHECK:STDERR:            ^~~~
  // CHECK:STDERR:
  var name name: C;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:22: error: found generic binding inside `var` pattern [NonRegularBindingInVarDecl]
  // CHECK:STDERR:   var generic name: C;
  // CHECK:STDERR:                      ^
  // CHECK:STDERR:
  var generic name: C;
}

// CHECK:STDOUT: - filename: fail_var_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │ ╭─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'name'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │ ╭─InvalidParse ':' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │ ╭─InvalidParse ':' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'name'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'name'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';' has_error
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'name'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';' has_error
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─SelfValueName 'self'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameExpr 'name'
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableDecl ';' has_error
// CHECK:STDOUT:   │ │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'name'
// CHECK:STDOUT:   │ │ │   ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │   ├─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':' has_error
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
