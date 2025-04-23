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
  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:6: error: expected identifier in field declaration [ExpectedFieldIdentifier]
  // CHECK:STDERR:   var;
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  var;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:11: error: expected `:` in field declaration [ExpectedFieldColon]
  // CHECK:STDERR:   var name;
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  var name;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:7: error: expected identifier in field declaration [ExpectedFieldIdentifier]
  // CHECK:STDERR:   var :;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  var :;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:7: error: expected identifier in field declaration [ExpectedFieldIdentifier]
  // CHECK:STDERR:   var : C;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  var : C;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:12: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   var name:;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  var name:;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:12: error: expected `:` in field declaration [ExpectedFieldColon]
  // CHECK:STDERR:   var name C;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  var name C;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:15: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   var name: C C;
  // CHECK:STDERR:               ^
  // CHECK:STDERR:
  var name: C C;

  // CHECK:STDERR: fail_var_name.carbon:[[@LINE+4]]:7: error: expected identifier in field declaration [ExpectedFieldIdentifier]
  // CHECK:STDERR:   var self: C;
  // CHECK:STDERR:       ^~~~
  // CHECK:STDERR:
  var self: C;
}

// CHECK:STDOUT: - filename: fail_var_name.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierNameNotBeforeParams', text: 'C'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'name'},
// CHECK:STDOUT:           {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'FieldNameAndType', text: ':', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:           {kind: 'IdentifierNameNotBeforeParams', text: 'name'},
// CHECK:STDOUT:           {kind: 'IdentifierNameExpr', text: 'C'},
// CHECK:STDOUT:         {kind: 'FieldNameAndType', text: ':', subtree_size: 3},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'FieldIntroducer', text: 'var'},
// CHECK:STDOUT:       {kind: 'FieldDecl', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 26},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
