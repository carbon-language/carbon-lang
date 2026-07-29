// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/function/declaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/function/declaration.carbon

// --- basic.carbon

fn F();

// --- ref.carbon

fn foo(ref a: i32);

// --- implicit_empty.carbon

fn foo[]();

// --- implicit_params.carbon

fn foo[a: i32, b: i32]();

// --- no_params.carbon

fn foo {}

// --- params.carbon

fn foo(a: i32, b: i32);

// --- with_return_type.carbon

fn foo() -> u32;

// --- impl_fn.carbon

override fn F();
abstract override fn G();
override abstract fn H();
private override default fn I();

// --- fail_identifier_instead_of_sig.carbon

// CHECK:STDERR: fail_identifier_instead_of_sig.carbon:[[@LINE+4]]:8: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: fn foo bar;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
fn foo bar;

// --- fail_missing_implicit_close.carbon

// CHECK:STDERR: fail_missing_implicit_close.carbon:[[@LINE+8]]:7: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fn Div[();
// CHECK:STDERR:       ^
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_implicit_close.carbon:[[@LINE+4]]:7: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR: fn Div[();
// CHECK:STDERR:       ^
// CHECK:STDERR:
fn Div[();

// --- fail_missing_name.carbon

// CHECK:STDERR: fail_missing_name.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn ();
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn ();

// --- fail_no_sig_or_semi.carbon

fn foo
// CHECK:STDERR: fail_no_sig_or_semi.carbon:[[@LINE+4]]:1: error: `fn` declarations must either end with a `;` or have a `{ ... }` block for a definition [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_only_fn_and_semi.carbon

// CHECK:STDERR: fail_only_fn_and_semi.carbon:[[@LINE+4]]:3: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn;
// CHECK:STDERR:   ^
// CHECK:STDERR:
fn;

// --- fail_repeated_fn_and_semi.carbon

// CHECK:STDERR: fail_repeated_fn_and_semi.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn fn;
// CHECK:STDERR:    ^~
// CHECK:STDERR:
fn fn;

// --- fail_skip_indented_newline_until_outdent.carbon

  // CHECK:STDERR: fail_skip_indented_newline_until_outdent.carbon:[[@LINE+4]]:6: error: `fn` introducer should be followed by a name [ExpectedDeclName]
  // CHECK:STDERR:   fn (x,
  // CHECK:STDERR:      ^
  // CHECK:STDERR:
  fn (x,
      y,
      z)
fn F();

// --- fail_skip_indented_newline_with_semi.carbon

// CHECK:STDERR: fail_skip_indented_newline_with_semi.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn (x,
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn (x,
    y,
    z);
fn F();

// --- fail_skip_indented_newline_without_semi.carbon

// CHECK:STDERR: fail_skip_indented_newline_without_semi.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn (x,
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn (x,
    y,
    z)
fn F();

// --- fail_skip_to_newline_without_semi.carbon

// CHECK:STDERR: fail_skip_to_newline_without_semi.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn ()
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn ()
fn F();

// --- fail_skip_without_semi_to_curly.carbon

// CHECK:STDERR: fail_skip_without_semi_to_curly.carbon:[[@LINE+4]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: struct X { fn () }
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
struct X { fn () }
fn F();

// --- with_identifier_as_param.carbon

fn foo(bar);

// --- fail_without_name_and_many_tokens_in_params.carbon

// CHECK:STDERR: fail_without_name_and_many_tokens_in_params.carbon:[[@LINE+4]]:4: error: `fn` introducer should be followed by a name [ExpectedDeclName]
// CHECK:STDERR: fn (a tokens c d e f g h i j k l m n o p q r s t u v w x y z);
// CHECK:STDERR:    ^
// CHECK:STDERR:
fn (a tokens c d e f g h i j k l m n o p q r s t u v w x y z);

// --- form_parameter.carbon

fn FormParam(x:? Form);
fn ComplexFormParam(x:? X.Y(Z));

// --- fail_ref_on_form_parameter.carbon

// CHECK:STDERR: fail_ref_on_form_parameter.carbon:[[@LINE+4]]:6: error: `ref` is only allowed on a runtime binding [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR: fn F(ref x:? Form);
// CHECK:STDERR:      ^~~
// CHECK:STDERR:
fn F(ref x:? Form);

// --- fail_form_within_var.carbon

// CHECK:STDERR: fail_form_within_var.carbon:[[@LINE+4]]:26: error: found `:?` binding inside `var` pattern [NonRegularBindingInVarDecl]
// CHECK:STDERR: fn FormInVar(var x:? Form);
// CHECK:STDERR:                          ^
// CHECK:STDERR:
fn FormInVar(var x:? Form);

// --- return_form.carbon

fn SimpleReturnForm() ->? Form;
fn ComplexReturnForm() ->? X.Y(Z);

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ╭─RefBindingName 'ref'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: implicit_empty.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: implicit_params.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ImplicitParamListStart '['
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ ├─CompileTimeBindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─CompileTimeBindingPattern ':'
// CHECK:STDOUT:   │ ├─ImplicitParamList ']'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: no_params.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'foo'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: params.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'a'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ │ ├─PatternListComma ','
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'b'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │ │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ├─LetBindingPattern ':'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: with_return_type.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ╭─UnsignedIntTypeLiteral 'u32'
// CHECK:STDOUT:   │ ├─ReturnType '->'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: impl_fn.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─OverrideModifier 'override'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─AbstractModifier 'abstract'
// CHECK:STDOUT:   │ ├─OverrideModifier 'override'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'G'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─OverrideModifier 'override'
// CHECK:STDOUT:   │ ├─AbstractModifier 'abstract'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'H'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─PrivateModifier 'private'
// CHECK:STDOUT:   │ ├─OverrideModifier 'override'
// CHECK:STDOUT:   │ ├─DefaultModifier 'default'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'I'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_identifier_instead_of_sig.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'foo'
// CHECK:STDOUT:   ├─FunctionDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_implicit_close.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Div'
// CHECK:STDOUT:   ├─FunctionDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_sig_or_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'foo'
// CHECK:STDOUT:   ├─FunctionDecl 'fn' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_only_fn_and_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse ';' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_repeated_fn_and_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse 'fn' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_skip_indented_newline_until_outdent.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl ')' ❌
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_skip_indented_newline_with_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';' ❌
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_skip_indented_newline_without_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl ')' ❌
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_skip_to_newline_without_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl ')' ❌
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_skip_without_semi_to_curly.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InvalidParseStart 'struct' ❌
// CHECK:STDOUT:   ├─InvalidParseSubtree '}' ❌
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: with_identifier_as_param.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'foo'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'bar'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_without_name_and_many_tokens_in_params.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─InvalidParse '(' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: form_parameter.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'FormParam'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':?'
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'Form'
// CHECK:STDOUT:   │ │ ├─FormBindingPattern ':?'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'ComplexFormParam'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':?'
// CHECK:STDOUT:   │ │ │ │     ╭─IdentifierNameExpr 'X'
// CHECK:STDOUT:   │ │ │ │     ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ │ │ │   ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │ │ │ ╭─CallExprStart '('
// CHECK:STDOUT:   │ │ │ │ ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ │ │ ├─CallExpr ')'
// CHECK:STDOUT:   │ │ ├─FormBindingPattern ':?'
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_ref_on_form_parameter.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │ ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │ ├─BindingPatternTypeStart ':?' ❌
// CHECK:STDOUT:   │ │ │ ├─IdentifierNameExpr 'Form'
// CHECK:STDOUT:   │ │ ├─FormBindingPattern ':?' ❌
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_form_within_var.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'FormInVar'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │ │   ├─BindingPatternTypeStart ':?'
// CHECK:STDOUT:   │ │ │   ├─IdentifierNameExpr 'Form'
// CHECK:STDOUT:   │ │ │ ╭─VarBindingPattern ':?' ❌
// CHECK:STDOUT:   │ │ ├─VariablePattern 'var' ❌
// CHECK:STDOUT:   │ ├─ExplicitParamList ')' ❌
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: return_form.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'SimpleReturnForm'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'Form'
// CHECK:STDOUT:   │ ├─ReturnForm '->?'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'ComplexReturnForm'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │       ╭─IdentifierNameExpr 'X'
// CHECK:STDOUT:   │ │       ├─IdentifierNameNotBeforeSignature 'Y'
// CHECK:STDOUT:   │ │     ╭─MemberAccessExpr '.'
// CHECK:STDOUT:   │ │   ╭─CallExprStart '('
// CHECK:STDOUT:   │ │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ │ ╭─CallExpr ')'
// CHECK:STDOUT:   │ ├─ReturnForm '->?'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
