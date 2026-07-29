// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/fail_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/fail_impl.carbon

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:10: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl foo bar;
// CHECK:STDERR:          ^~~
// CHECK:STDERR:
impl foo bar;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:12: error: expected expression [ExpectedExpr]
// CHECK:STDERR: impl i32 as;
// CHECK:STDERR:            ^
// CHECK:STDERR:
impl i32 as;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:18: error: `impl` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: impl bool as bar unexpected;
// CHECK:STDERR:                  ^~~~~~~~~~
// CHECK:STDERR:
impl bool as bar unexpected;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:6: error: expected expression [ExpectedExpr]
// CHECK:STDERR: impl return as A;
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR:
impl return as A;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:6: error: expected expression [ExpectedExpr]
// CHECK:STDERR: impl return B;
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR:
impl return B;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+8]]:13: error: expected `[` after `forall` in `impl` declaration [ImplExpectedAfterForall]
// CHECK:STDERR: impl forall f32;
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:16: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl forall f32;
// CHECK:STDERR:                ^
// CHECK:STDERR:
impl forall f32;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:19: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl forall [] u32;
// CHECK:STDERR:                   ^
// CHECK:STDERR:
impl forall [] u32;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:25: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl forall [invalid] i8;
// CHECK:STDERR:                         ^
// CHECK:STDERR:
impl forall [invalid] i8;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:13: error: expected `[` after `forall` in `impl` declaration [ImplExpectedAfterForall]
// CHECK:STDERR: impl forall f16 as Quux;
// CHECK:STDERR:             ^~~
// CHECK:STDERR:
impl forall f16 as Quux;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:26: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl forall [T: type] str;
// CHECK:STDERR:                          ^
// CHECK:STDERR:
impl forall [T: type] str;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:25: error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR: impl forall [T: type] T missing_as;
// CHECK:STDERR:                         ^~~~~~~~~~
// CHECK:STDERR:
impl forall [T: type] T missing_as;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:38: error: `impl` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: impl forall [T: type] T as Interface extra;
// CHECK:STDERR:                                      ^~~~~
// CHECK:STDERR:
impl forall [T: type] T as Interface extra;

// CHECK:STDERR: fail_impl.carbon:[[@LINE+4]]:5: error: expected expression [ExpectedExpr]
// CHECK:STDERR: impl;
// CHECK:STDERR:     ^
// CHECK:STDERR:
impl;

impl

// CHECK:STDERR: fail_impl.carbon:[[@LINE+89]]:26: error: expected expression [ExpectedExpr]
// CHECK:STDERR: // CHECK:STDOUT:   (root)
// CHECK:STDERR:                          ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_impl.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─InvalidParse ';' ❌
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'bar'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse 'return' ❌
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse 'return' ❌
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ ├─InvalidParse 'f32' ❌
// CHECK:STDOUT:   │ ├─FloatTypeLiteral 'f32'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─UnsignedIntTypeLiteral 'u32'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'invalid'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─IntTypeLiteral 'i8'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ ├─InvalidParse 'f16' ❌
// CHECK:STDOUT:   │ │ ╭─FloatTypeLiteral 'f16'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'Quux'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'Interface'
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse ';' ❌
// CHECK:STDOUT:   ├─ImplDecl ';' ❌
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse '' ❌
// CHECK:STDOUT:   ├─ImplDecl 'impl' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
