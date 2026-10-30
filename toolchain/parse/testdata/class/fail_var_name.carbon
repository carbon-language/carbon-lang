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
  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+6]]:6
  // CHECK:STDERR: ->     │   var;
  // CHECK:STDERR:        ·      ┳
  // CHECK:STDERR:        ·      ┗━━┫ error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var;

  var name;

  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+6]]:7
  // CHECK:STDERR: ->     │   var :;
  // CHECK:STDERR:        ·       ┳
  // CHECK:STDERR:        ·       ┗━━┫ error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var :;

  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+6]]:7
  // CHECK:STDERR: ->     │   var : C;
  // CHECK:STDERR:        ·       ┳
  // CHECK:STDERR:        ·       ┗━━┫ error: expected pattern [ExpectedPattern]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var : C;

  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+6]]:12
  // CHECK:STDERR: ->     │   var name:;
  // CHECK:STDERR:        ·            ┳
  // CHECK:STDERR:        ·            ┗━━┫ error: expected expression [ExpectedExpr]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var name:;

  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+6]]:12
  // CHECK:STDERR: ->     │   var name C;
  // CHECK:STDERR:        ·            ┳
  // CHECK:STDERR:        ·            ┗━━┫ error: `var` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var name C;

  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+7]]:15
  // CHECK:STDERR: ->     │   var name: C C;
  // CHECK:STDERR:        ·               ┳
  // CHECK:STDERR:        ·               ┗━━┫ error: `var` declarations must end with a `;`
  // CHECK:STDERR:        ·                  ┃        [ExpectedDeclSemi]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var name: C C;

  var self: C;

  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+7]]:12
  // CHECK:STDERR: ->     │   var name name: C;
  // CHECK:STDERR:        ·            ━━┳━
  // CHECK:STDERR:        ·              ┗━━┫ error: `var` declarations must end with a `;`
  // CHECK:STDERR:        ·                 ┃        [ExpectedDeclSemi]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var name name: C;

  // CHECK:STDERR:        ╭─┤ fail_var_name.carbon:[[@LINE+7]]:22
  // CHECK:STDERR: ->     │   var generic name: C;
  // CHECK:STDERR:        ·                      ┳
  // CHECK:STDERR:        ·                      ┗━━┫ error: found generic binding inside `var` pattern
  // CHECK:STDERR:        ·                         ┃        [NonRegularBindingInVarDecl]
  // CHECK:STDERR:        ╰────
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
