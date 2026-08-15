// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/fail_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/fail_base.carbon

class A {
  // CHECK:STDERR: fail_base.carbon:[[@LINE+4]]:7: error: `class` or `:` expected after `base` [ExpectedAfterBase]
  // CHECK:STDERR:   base;
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  base;

  // CHECK:STDERR: fail_base.carbon:[[@LINE+4]]:8: error: `class` or `:` expected after `base` [ExpectedAfterBase]
  // CHECK:STDERR:   base calss X {}
  // CHECK:STDERR:        ^~~~~
  // CHECK:STDERR:
  base calss X {}

  // CHECK:STDERR: fail_base.carbon:[[@LINE+4]]:9: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   base: ;
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  base: ;

  base: { }
  // We should resume parsing here after the previous error.
  // CHECK:STDERR: fail_base.carbon:[[@LINE+4]]:3: error: `base` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   var n: i32;
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  var n: i32;
}

base class Foo {}

class B {
  // CHECK:STDERR: fail_base.carbon:[[@LINE+4]]:15: error: `class` or `:` expected after `base` [ExpectedAfterBase]
  // CHECK:STDERR:   extend base Foo;
  // CHECK:STDERR:               ^~~
  // CHECK:STDERR:
  extend base Foo;
}

// CHECK:STDOUT: - filename: fail_base.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ ├─BaseDecl ';' has_error
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ ├─BaseDecl '}' has_error
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ │ ├─BaseColon ':'
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ ├─BaseDecl ';' has_error
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ │ ├─BaseColon ':'
// CHECK:STDOUT:   │ │ │ ╭─StructLiteralStart '{'
// CHECK:STDOUT:   │ │ ├─StructLiteral '}'
// CHECK:STDOUT:   │ ├─BaseDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─BaseModifier 'base'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─BaseIntroducer 'base'
// CHECK:STDOUT:   │ │ ├─ExtendModifier 'extend'
// CHECK:STDOUT:   │ │ ├─BaseColon 'Foo' has_error
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'Foo'
// CHECK:STDOUT:   │ ├─BaseDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
