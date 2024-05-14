// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class A {
  // CHECK:STDERR: fail_base.carbon:[[@LINE+4]]:7: ERROR: `class` or `:` expected after `base`.
  // CHECK:STDERR:   base;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  base;

  // CHECK:STDERR: fail_base.carbon:[[@LINE+4]]:9: ERROR: Expected expression.
  // CHECK:STDERR:   base: ;
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  base: ;

  base: { }
  // We should resume parsing here after the previous error.
  // CHECK:STDERR: fail_base.carbon:[[@LINE+7]]:3: ERROR: `base` declarations must end with a `;`.
  // CHECK:STDERR:   var n: i32;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_base.carbon:[[@LINE+3]]:7: ERROR: Unrecognized declaration introducer.
  // CHECK:STDERR:   var n: i32;
  // CHECK:STDERR:       ^
  var n: i32;
}

// CHECK:STDOUT: - filename: fail_base.carbon
// CHECK:STDOUT:   parse_tree: [
// CHECK:STDOUT:     {kind: 'FileStart', text: ''},
// CHECK:STDOUT:         {kind: 'ClassIntroducer', text: 'class'},
// CHECK:STDOUT:         {kind: 'IdentifierName', text: 'A'},
// CHECK:STDOUT:       {kind: 'ClassDefinitionStart', text: '{', subtree_size: 3},
// CHECK:STDOUT:         {kind: 'InvalidParseStart', text: ';', has_error: yes},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: 'base', has_error: yes},
// CHECK:STDOUT:       {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 3},
// CHECK:STDOUT:         {kind: 'BaseIntroducer', text: 'base'},
// CHECK:STDOUT:         {kind: 'BaseColon', text: ':'},
// CHECK:STDOUT:         {kind: 'InvalidParse', text: ';', has_error: yes},
// CHECK:STDOUT:       {kind: 'BaseDecl', text: ';', has_error: yes, subtree_size: 4},
// CHECK:STDOUT:         {kind: 'BaseIntroducer', text: 'base'},
// CHECK:STDOUT:         {kind: 'BaseColon', text: ':'},
// CHECK:STDOUT:           {kind: 'StructLiteralStart', text: '{'},
// CHECK:STDOUT:         {kind: 'StructLiteral', text: '}', subtree_size: 2},
// CHECK:STDOUT:       {kind: 'BaseDecl', text: 'var', has_error: yes, subtree_size: 5},
// CHECK:STDOUT:         {kind: 'InvalidParseStart', text: 'n', has_error: yes},
// CHECK:STDOUT:       {kind: 'InvalidParseSubtree', text: ';', has_error: yes, subtree_size: 2},
// CHECK:STDOUT:     {kind: 'ClassDefinition', text: '}', subtree_size: 18},
// CHECK:STDOUT:     {kind: 'FileEnd', text: ''},
// CHECK:STDOUT:   ]
