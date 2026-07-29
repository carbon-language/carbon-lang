// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/alias/fail_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/alias/fail_syntax.carbon

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:7: error: `alias` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: alias = b;
// CHECK:STDERR:       ^
// CHECK:STDERR:
alias = b;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:9: error: `alias` requires a `=` for the source [ExpectedAliasInitializer]
// CHECK:STDERR: alias a b;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias a b;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:10: error: expected expression [ExpectedExpr]
// CHECK:STDERR: alias a =;
// CHECK:STDERR:          ^
// CHECK:STDERR:
alias a =;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:8: error: `alias` requires a `=` for the source [ExpectedAliasInitializer]
// CHECK:STDERR: alias a;
// CHECK:STDERR:        ^
// CHECK:STDERR:
alias a;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:7: error: `alias` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: alias =;
// CHECK:STDERR:       ^
// CHECK:STDERR:
alias =;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:9: error: `alias` requires a `=` for the source [ExpectedAliasInitializer]
// CHECK:STDERR: alias a b c = d;
// CHECK:STDERR:         ^
// CHECK:STDERR:
alias a b c = d;

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:13: error: `alias` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: alias a = b c d;
// CHECK:STDERR:             ^
// CHECK:STDERR:
alias a = b c d;

fn F() {
  // CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:8: error: `alias` introducer should be followed by a name [ExpectedDeclName]
  // CHECK:STDERR:   alias;
  // CHECK:STDERR:        ^
  // CHECK:STDERR:
  alias;

  // CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:12: error: expected expression [ExpectedExpr]
  // CHECK:STDERR:   alias a =;
  // CHECK:STDERR:            ^
  // CHECK:STDERR:
  alias a =;

  alias a =
// CHECK:STDERR: fail_syntax.carbon:[[@LINE+4]]:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

alias a = b

// CHECK:STDERR: fail_syntax.carbon:[[@LINE+56]]:26: error: `alias` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: // CHECK:STDOUT:   (root)
// CHECK:STDERR:                          ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_syntax.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─InvalidParse '=' ❌
// CHECK:STDOUT:   ├─Alias ';' ❌
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   ├─Alias ';' ❌
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ ├─InvalidParse ';' ❌
// CHECK:STDOUT:   ├─Alias ';' ❌
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   ├─Alias ';' ❌
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─InvalidParse '=' ❌
// CHECK:STDOUT:   ├─Alias ';' ❌
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   ├─Alias ';' ❌
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   ├─Alias ';' ❌
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' ❌
// CHECK:STDOUT:   │ ├─Alias ';' ❌
// CHECK:STDOUT:   │ │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' ❌
// CHECK:STDOUT:   │ ├─Alias ';' ❌
// CHECK:STDOUT:   │ │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ │ ├─InvalidParse '}' ❌
// CHECK:STDOUT:   │ ├─Alias '=' ❌
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   │ ╭─AliasIntroducer 'alias'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ ├─AliasInitializer '='
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   ├─Alias 'b' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
