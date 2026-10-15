// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/class/friend.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/class/friend.carbon

// --- friend.carbon

class Foo {
  friend Bar;
}

// --- fail_todo_qualified.carbon

class Baz {
  // CHECK:STDERR: fail_todo_qualified.carbon:[[@LINE+4]]:11: error: `friend` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   friend P.Q;
  // CHECK:STDERR:           ^
  // CHECK:STDERR:
  friend P.Q;
}

// --- fail_bad_syntax.carbon

class A {
  // CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:9: error: expected identifier for friend declaration [ExpectedIdentifierForFriend]
  // CHECK:STDERR:   friend;
  // CHECK:STDERR:         ^
  // CHECK:STDERR:
  friend;
}

class B {
  // CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:12: error: `friend` declarations must end with a `;` [ExpectedDeclSemi]
  // CHECK:STDERR:   friend X extra_token;
  // CHECK:STDERR:            ^~~~~~~~~~~
  // CHECK:STDERR:
  friend X extra_token;
}

class C {
  friend
// CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:1: error: expected identifier for friend declaration [ExpectedIdentifierForFriend]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

class D {
  friend X
// CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:1: error: `friend` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

class E {
  // CHECK:STDERR: fail_bad_syntax.carbon:[[@LINE+4]]:10: error: expected identifier for friend declaration [ExpectedIdentifierForFriend]
  // CHECK:STDERR:   friend class;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  friend class;
}

// CHECK:STDOUT: - filename: friend.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FriendIntroducer 'friend'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'Bar'
// CHECK:STDOUT:   │ ├─FriendDecl ';'
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_todo_qualified.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'Baz'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FriendIntroducer 'friend'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'P'
// CHECK:STDOUT:   │ ├─FriendDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_bad_syntax.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'A'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FriendIntroducer 'friend'
// CHECK:STDOUT:   │ ├─FriendDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'B'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FriendIntroducer 'friend'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'X'
// CHECK:STDOUT:   │ ├─FriendDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'C'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FriendIntroducer 'friend'
// CHECK:STDOUT:   │ ├─FriendDecl 'friend' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'D'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FriendIntroducer 'friend'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'X'
// CHECK:STDOUT:   │ ├─FriendDecl 'X' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   │   ╭─ClassIntroducer 'class'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'E'
// CHECK:STDOUT:   │ ╭─ClassDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─FriendIntroducer 'friend'
// CHECK:STDOUT:   │ ├─FriendDecl ';' has_error
// CHECK:STDOUT:   ├─ClassDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
