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
  // CHECK:STDERR:   var self: C;
  // CHECK:STDERR:       ^~~~
  // CHECK:STDERR:
  var self: C;

  var :;

  var : C;

  var name:;

  var name C;

  var name: C C;
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
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 10},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
