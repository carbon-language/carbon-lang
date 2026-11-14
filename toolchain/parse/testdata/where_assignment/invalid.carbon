// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_assignment/invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_assignment/invalid.carbon

// --- fail_self.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ fail_self.carbon:[[@LINE+7]]:8
  // CHECK:STDERR: ->     │   where Self = 1;
  // CHECK:STDERR:        ·        ┳
  // CHECK:STDERR:        ·        ┗━━┫ error: expected identifier in `where` assignment declaration
  // CHECK:STDERR:        ·           ┃        [WhereAssignmentMissingIdentifier]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Self = 1;
}

// --- fail_keyword.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ fail_keyword.carbon:[[@LINE+7]]:8
  // CHECK:STDERR: ->     │   where and = 1;
  // CHECK:STDERR:        ·        ┳
  // CHECK:STDERR:        ·        ┗━━┫ error: expected identifier in `where` assignment declaration
  // CHECK:STDERR:        ·           ┃        [WhereAssignmentMissingIdentifier]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where and = 1;
}

// --- fail_postfix_after_lhs.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ fail_postfix_after_lhs.carbon:[[@LINE+7]]:11
  // CHECK:STDERR: ->     │   where Z1* = 1;
  // CHECK:STDERR:        ·           ┳
  // CHECK:STDERR:        ·           ┗━━┫ error: missing `=` in `where` assignment
  // CHECK:STDERR:        ·              ┃        [WhereAssignmentMissingEquals]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1* = 1;
}

// --- fail_binary_op_after_lhs.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ fail_binary_op_after_lhs.carbon:[[@LINE+7]]:11
  // CHECK:STDERR: ->     │   where Z1 + Z2 = 1;
  // CHECK:STDERR:        ·           ┳
  // CHECK:STDERR:        ·           ┗━━┫ error: missing `=` in `where` assignment
  // CHECK:STDERR:        ·              ┃        [WhereAssignmentMissingEquals]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 + Z2 = 1;
}

// --- fail_designator.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ fail_designator.carbon:[[@LINE+7]]:8
  // CHECK:STDERR: ->     │   where .Z1 = 1;
  // CHECK:STDERR:        ·        ┳
  // CHECK:STDERR:        ·        ┗━━┫ error: expected identifier in `where` assignment declaration
  // CHECK:STDERR:        ·           ┃        [WhereAssignmentMissingIdentifier]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where .Z1 = 1;
}

// --- fail_lhs_in_parens.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ fail_lhs_in_parens.carbon:[[@LINE+7]]:8
  // CHECK:STDERR: ->     │   where (Z1) = 1;
  // CHECK:STDERR:        ·        ┳
  // CHECK:STDERR:        ·        ┗━━┫ error: expected identifier in `where` assignment declaration
  // CHECK:STDERR:        ·           ┃        [WhereAssignmentMissingIdentifier]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where (Z1) = 1;
}

// --- fail_multiple_assignments.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ fail_multiple_assignments.carbon:[[@LINE+7]]:15
  // CHECK:STDERR: ->     │   where Z1 = 1 and Z2 = 2;
  // CHECK:STDERR:        ·               ┳
  // CHECK:STDERR:        ·               ┗━━┫ error: `where` declarations must end with a `;`
  // CHECK:STDERR:        ·                  ┃        [ExpectedDeclSemi]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = 1 and Z2 = 2;
}

// CHECK:STDOUT: - filename: fail_self.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_keyword.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_postfix_after_lhs.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'Z1'
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_binary_op_after_lhs.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'Z1'
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_designator.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_lhs_in_parens.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_multiple_assignments.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─WhereAssignmentIntroducer 'where'
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'Z1'
// CHECK:STDOUT:   │ │ │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │ ├─WhereAssignmentOperator '='
// CHECK:STDOUT:   │ ├─WhereAssignmentDecl ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
