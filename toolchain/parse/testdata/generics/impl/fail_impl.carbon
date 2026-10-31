// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/impl/fail_impl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/impl/fail_impl.carbon

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+6]]:10
// CHECK:STDERR: ->     │ impl foo bar;
// CHECK:STDERR:        ·          ━┳━
// CHECK:STDERR:        ·           ┗━━┫ error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl foo bar;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+6]]:12
// CHECK:STDERR: ->     │ impl i32 as;
// CHECK:STDERR:        ·            ┳
// CHECK:STDERR:        ·            ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl i32 as;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+8]]:18
// CHECK:STDERR: ->     │ impl bool as bar unexpected;
// CHECK:STDERR:        ·                  ┳━━━━━━━━━
// CHECK:STDERR:        ·                  ┗━━┫ error: `impl` declarations must either end with a `;` or
// CHECK:STDERR:        ·                     ┃        have a `{ ... }` block for a definition
// CHECK:STDERR:        ·                     ┃        [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl bool as bar unexpected;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+6]]:6
// CHECK:STDERR: ->     │ impl return as A;
// CHECK:STDERR:        ·      ━━━┳━━
// CHECK:STDERR:        ·         ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl return as A;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+6]]:6
// CHECK:STDERR: ->     │ impl return B;
// CHECK:STDERR:        ·      ━━━┳━━
// CHECK:STDERR:        ·         ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl return B;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+13]]:13
// CHECK:STDERR: ->     │ impl forall f32;
// CHECK:STDERR:        ·             ━┳━
// CHECK:STDERR:        ·              ┗━━┫ error: expected `[` after `forall` in `impl` declaration
// CHECK:STDERR:        ·                 ┃        [ImplExpectedAfterForall]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+6]]:16
// CHECK:STDERR: ->     │ impl forall f32;
// CHECK:STDERR:        ·                ┳
// CHECK:STDERR:        ·                ┗━━┫ error: expected `as` in `impl` declaration [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl forall f32;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+7]]:19
// CHECK:STDERR: ->     │ impl forall [] u32;
// CHECK:STDERR:        ·                   ┳
// CHECK:STDERR:        ·                   ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                      ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl forall [] u32;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+7]]:25
// CHECK:STDERR: ->     │ impl forall [invalid] i8;
// CHECK:STDERR:        ·                         ┳
// CHECK:STDERR:        ·                         ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                            ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl forall [invalid] i8;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+7]]:13
// CHECK:STDERR: ->     │ impl forall f16 as Quux;
// CHECK:STDERR:        ·             ━┳━
// CHECK:STDERR:        ·              ┗━━┫ error: expected `[` after `forall` in `impl` declaration
// CHECK:STDERR:        ·                 ┃        [ImplExpectedAfterForall]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl forall f16 as Quux;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+7]]:26
// CHECK:STDERR: ->     │ impl forall [T: type] str;
// CHECK:STDERR:        ·                          ┳
// CHECK:STDERR:        ·                          ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                             ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl forall [T: type] str;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+7]]:25
// CHECK:STDERR: ->     │ impl forall [T: type] T missing_as;
// CHECK:STDERR:        ·                         ━━━━━┳━━━━
// CHECK:STDERR:        ·                              ┗━━┫ error: expected `as` in `impl` declaration
// CHECK:STDERR:        ·                                 ┃        [ImplExpectedAs]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl forall [T: type] T missing_as;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+10]]:38
// CHECK:STDERR: ->     │ impl forall [T: type] T as Interface extra;
// CHECK:STDERR:        ·                                      ┳━━━━
// CHECK:STDERR:        ·                                      ┗━━┫ error: `impl` declarations must either
// CHECK:STDERR:        ·                                         ┃        end with a `;` or have a
// CHECK:STDERR:        ·                                         ┃        `{ ... }` block for a
// CHECK:STDERR:        ·                                         ┃        definition
// CHECK:STDERR:        ·                                         ┃        [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl forall [T: type] T as Interface extra;

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+6]]:5
// CHECK:STDERR: ->     │ impl;
// CHECK:STDERR:        ·     ┳
// CHECK:STDERR:        ·     ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl;

impl

// CHECK:STDERR:        ╭─┤ fail_impl.carbon:[[@LINE+91]]:26
// CHECK:STDERR: ->     │ // CHECK:STDOUT:   (root)
// CHECK:STDERR:        ·                          ┳
// CHECK:STDERR:        ·                          ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_impl.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'foo'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ╭─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ │ ╭─BoolTypeLiteral 'bool'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'bar'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse 'return' has_error
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse 'return' has_error
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ ├─InvalidParse 'f32' has_error
// CHECK:STDOUT:   │ ├─FloatTypeLiteral 'f32'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─UnsignedIntTypeLiteral 'u32'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'invalid'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─IntTypeLiteral 'i8'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ ├─InvalidParse 'f16' has_error
// CHECK:STDOUT:   │ │ ╭─FloatTypeLiteral 'f16'
// CHECK:STDOUT:   │ ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'Quux'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─StringTypeLiteral 'str'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─Forall 'forall'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'T'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─TypeTypeLiteral 'type'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
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
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   ├─ImplDecl ';' has_error
// CHECK:STDOUT:   │ ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │ ├─InvalidParse '' has_error
// CHECK:STDOUT:   ├─ImplDecl 'impl' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
