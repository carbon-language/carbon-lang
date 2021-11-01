// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
//
// The below output should only be used for basic validation -- changes are
// okay.
//
// AUTOUPDATE: executable_semantics --trace %s
// CHECK: Starting parse
// CHECK: Entering state 0
// CHECK: Reading a token: Next token is token PACKAGE (371.1-7: )
// CHECK: Shifting token PACKAGE (371.1-7: )
// CHECK: Entering state 1
// CHECK: Reading a token: Next token is token identifier (371.9-31: )
// CHECK: Shifting token identifier (371.9-31: )
// CHECK: Entering state 4
// CHECK: Reading a token: Next token is token API (371.33-35: )
// CHECK: Reducing stack by rule 6 (line 262):
// CHECK: -> $$ = nterm optional_library_path (371.32: )
// CHECK: Stack now 4 1 0
// CHECK: Entering state 8
// CHECK: Next token is token API (371.33-35: )
// CHECK: Shifting token API (371.33-35: )
// CHECK: Entering state 13
// CHECK: Reducing stack by rule 8 (line 267):
// CHECK:    $1 = token API (371.33-35: )
// CHECK: -> $$ = nterm api_or_impl (371.33-35: )
// CHECK: Stack now 8 4 1 0
// CHECK: Entering state 15
// CHECK: Reading a token: Next token is token SEMICOLON (371.36: )
// CHECK: Shifting token SEMICOLON (371.36: )
// CHECK: Entering state 23
// CHECK: Reducing stack by rule 2 (line 244):
// CHECK:    $1 = token PACKAGE (371.1-7: )
// CHECK:    $2 = token identifier (371.9-31: )
// CHECK:    $3 = nterm optional_library_path (371.32: )
// CHECK:    $4 = nterm api_or_impl (371.33-35: )
// CHECK:    $5 = token SEMICOLON (371.36: )
// CHECK: -> $$ = nterm package_directive (371.1-36: )
// CHECK: Stack now 0
// CHECK: Entering state 3
// CHECK: Reducing stack by rule 4 (line 253):
// CHECK: -> $$ = nterm import_directives (371.37: )
// CHECK: Stack now 3 0
// CHECK: Entering state 6
// CHECK: Reading a token: Next token is token FN (373.1-2: )
// CHECK: Reducing stack by rule 128 (line 734):
// CHECK: -> $$ = nterm declaration_list (371.37: )
// CHECK: Stack now 6 3 0
// CHECK: Entering state 11
// CHECK: Next token is token FN (373.1-2: )
// CHECK: Shifting token FN (373.1-2: )
// CHECK: Entering state 19
// CHECK: Reading a token: Next token is token identifier (373.4-7: )
// CHECK: Shifting token identifier (373.4-7: )
// CHECK: Entering state 27
// CHECK: Reading a token: Next token is token LEFT_PARENTHESIS (373.8: )
// CHECK: Reducing stack by rule 108 (line 652):
// CHECK: -> $$ = nterm deduced_params (373.8: )
// CHECK: Stack now 27 19 11 6 3 0
// CHECK: Entering state 34
// CHECK: Next token is token LEFT_PARENTHESIS (373.8: )
// CHECK: Shifting token LEFT_PARENTHESIS (373.8: )
// CHECK: Entering state 45
// CHECK: Reading a token: Next token is token RIGHT_PARENTHESIS (373.9: )
// CHECK: Shifting token RIGHT_PARENTHESIS (373.9: )
// CHECK: Entering state 91
// CHECK: Reducing stack by rule 72 (line 519):
// CHECK:    $1 = token LEFT_PARENTHESIS (373.8: )
// CHECK:    $2 = token RIGHT_PARENTHESIS (373.9: )
// CHECK: -> $$ = nterm maybe_empty_tuple_pattern (373.8-9: )
// CHECK: Stack now 34 27 19 11 6 3 0
// CHECK: Entering state 48
// CHECK: Reading a token: Next token is token ARROW (373.11-12: )
// CHECK: Shifting token ARROW (373.11-12: )
// CHECK: Entering state 96
// CHECK: Reading a token: Next token is token sized_type_literal (373.14-16: )
// CHECK: Shifting token sized_type_literal (373.14-16: )
// CHECK: Entering state 51
// CHECK: Reducing stack by rule 17 (line 287):
// CHECK:    $1 = token sized_type_literal (373.14-16: )
// CHECK: -> $$ = nterm expression (373.14-16: )
// CHECK: Stack now 96 48 34 27 19 11 6 3 0
// CHECK: Entering state 134
// CHECK: Reading a token: Next token is token LEFT_CURLY_BRACE (373.18: )
// CHECK: Reducing stack by rule 103 (line 629):
// CHECK:    $1 = token ARROW (373.11-12: )
// CHECK:    $2 = nterm expression (373.14-16: )
// CHECK: -> $$ = nterm return_type (373.11-16: )
// CHECK: Stack now 48 34 27 19 11 6 3 0
// CHECK: Entering state 97
// CHECK: Next token is token LEFT_CURLY_BRACE (373.18: )
// CHECK: Shifting token LEFT_CURLY_BRACE (373.18: )
// CHECK: Entering state 135
// CHECK: Reading a token: Next token is token RETURN (374.3-8: )
// CHECK: Shifting token RETURN (374.3-8: )
// CHECK: Entering state 172
// CHECK: Reading a token: Next token is token integer_literal (374.10: )
// CHECK: Shifting token integer_literal (374.10: )
// CHECK: Entering state 49
// CHECK: Reducing stack by rule 13 (line 279):
// CHECK:    $1 = token integer_literal (374.10: )
// CHECK: -> $$ = nterm expression (374.10: )
// CHECK: Stack now 172 135 97 48 34 27 19 11 6 3 0
// CHECK: Entering state 198
// CHECK: Reading a token: Next token is token SEMICOLON (374.11: )
// CHECK: Reducing stack by rule 96 (line 606):
// CHECK:    $1 = nterm expression (374.10: )
// CHECK: -> $$ = nterm return_expression (374.10: )
// CHECK: Stack now 172 135 97 48 34 27 19 11 6 3 0
// CHECK: Entering state 199
// CHECK: Next token is token SEMICOLON (374.11: )
// CHECK: Shifting token SEMICOLON (374.11: )
// CHECK: Entering state 213
// CHECK: Reducing stack by rule 85 (line 563):
// CHECK:    $1 = token RETURN (374.3-8: )
// CHECK:    $2 = nterm return_expression (374.10: )
// CHECK:    $3 = token SEMICOLON (374.11: )
// CHECK: -> $$ = nterm statement (374.3-11: )
// CHECK: Stack now 135 97 48 34 27 19 11 6 3 0
// CHECK: Entering state 177
// CHECK: Reading a token: Next token is token RIGHT_CURLY_BRACE (375.1: )
// CHECK: Reducing stack by rule 97 (line 611):
// CHECK: -> $$ = nterm statement_list (374.12: )
// CHECK: Stack now 177 135 97 48 34 27 19 11 6 3 0
// CHECK: Entering state 205
// CHECK: Reducing stack by rule 99 (line 616):
// CHECK:    $1 = nterm statement (374.3-11: )
// CHECK:    $2 = nterm statement_list (374.12: )
// CHECK: -> $$ = nterm nonempty_statement_list (374.3-11: )
// CHECK: Stack now 135 97 48 34 27 19 11 6 3 0
// CHECK: Entering state 180
// CHECK: Reducing stack by rule 98 (line 612):
// CHECK:    $1 = nterm nonempty_statement_list (374.3-11: )
// CHECK: -> $$ = nterm statement_list (374.3-11: )
// CHECK: Stack now 135 97 48 34 27 19 11 6 3 0
// CHECK: Entering state 179
// CHECK: Next token is token RIGHT_CURLY_BRACE (375.1: )
// CHECK: Shifting token RIGHT_CURLY_BRACE (375.1: )
// CHECK: Entering state 206
// CHECK: Reducing stack by rule 100 (line 620):
// CHECK:    $1 = token LEFT_CURLY_BRACE (373.18: )
// CHECK:    $2 = nterm statement_list (374.3-11: )
// CHECK:    $3 = token RIGHT_CURLY_BRACE (375.1: )
// CHECK: -> $$ = nterm block (373.18-375.1: )
// CHECK: Stack now 97 48 34 27 19 11 6 3 0
// CHECK: Entering state 137
// CHECK: Reducing stack by rule 110 (line 657):
// CHECK:    $1 = token FN (373.1-2: )
// CHECK:    $2 = token identifier (373.4-7: )
// CHECK:    $3 = nterm deduced_params (373.8: )
// CHECK:    $4 = nterm maybe_empty_tuple_pattern (373.8-9: )
// CHECK:    $5 = nterm return_type (373.11-16: )
// CHECK:    $6 = nterm block (373.18-375.1: )
// CHECK: -> $$ = nterm function_declaration (373.1-375.1: )
// CHECK: Stack now 11 6 3 0
// CHECK: Entering state 21
// CHECK: Reducing stack by rule 124 (line 723):
// CHECK:    $1 = nterm function_declaration (373.1-375.1: )
// CHECK: -> $$ = nterm declaration (373.1-375.1: )
// CHECK: Stack now 11 6 3 0
// CHECK: Entering state 22
// CHECK: Reducing stack by rule 129 (line 735):
// CHECK:    $1 = nterm declaration_list (371.37: )
// CHECK:    $2 = nterm declaration (373.1-375.1: )
// CHECK: -> $$ = nterm declaration_list (371.37-375.1: )
// CHECK: Stack now 6 3 0
// CHECK: Entering state 11
// CHECK: Reading a token: Next token is token END_OF_FILE (376.1: )
// CHECK: Reducing stack by rule 1 (line 235):
// CHECK:    $1 = nterm package_directive (371.1-36: )
// CHECK:    $2 = nterm import_directives (371.37: )
// CHECK:    $3 = nterm declaration_list (371.37-375.1: )
// CHECK: -> $$ = nterm input (371.1-375.1: )
// CHECK: Stack now 0
// CHECK: Entering state 2
// CHECK: Next token is token END_OF_FILE (376.1: )
// CHECK: Shifting token END_OF_FILE (376.1: )
// CHECK: Entering state 5
// CHECK: Cleanup: popping token END_OF_FILE (376.1: )
// CHECK: Cleanup: popping nterm input (371.1-375.1: )
// CHECK: ********** source program **********
// CHECK: fn Print (format_str: String) -> () {
// CHECK: {
// CHECK: return intrinsic_expression(print);
// CHECK-EMPTY:
// CHECK: }
// CHECK-EMPTY:
// CHECK: }
// CHECK: fn main () -> i32 {
// CHECK: {
// CHECK: return 0;
// CHECK-EMPTY:
// CHECK: }
// CHECK-EMPTY:
// CHECK: }
// CHECK: ********** type checking **********
// CHECK: checking pattern (format_str: String)
// CHECK: types:
// CHECK: values:
// CHECK: checking pattern format_str: String
// CHECK: types:
// CHECK: values:
// CHECK: checking pattern String
// CHECK: types:
// CHECK: values:
// CHECK: checking expression String
// CHECK: types:
// CHECK: values:
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern (format_str: String) (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern (format_str: String) (<intrinsic>:0) --->
// CHECK: --- step pattern () (<intrinsic>:0) --->
// CHECK: --- step exp () (<intrinsic>:0) --->
// CHECK: checking pattern ()
// CHECK: types: Print: fn (String) -> ()
// CHECK: values: Print: fun<Print>
// CHECK: --- step pattern () ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:373) --->
// CHECK: --- step pattern i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:373) --->
// CHECK: --- step exp i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:373) --->
// CHECK: checking pattern (format_str: String)
// CHECK: types: main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: checking pattern format_str: String
// CHECK: types: main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: checking pattern String
// CHECK: types: main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: checking expression String
// CHECK: types: main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern (format_str: String) (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern String (<intrinsic>:0) --->
// CHECK: --- step exp String (<intrinsic>:0) --->
// CHECK: --- step pattern format_str: String (<intrinsic>:0) --->
// CHECK: --- step pattern (format_str: String) (<intrinsic>:0) --->
// CHECK: --- step pattern () (<intrinsic>:0) --->
// CHECK: --- step exp () (<intrinsic>:0) --->
// CHECK: checking expression intrinsic_expression(print)
// CHECK: types: format_str: String, main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: checking pattern ()
// CHECK: types: main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: --- step pattern () ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:373) --->
// CHECK: --- step pattern i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:373) --->
// CHECK: --- step exp i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:373) --->
// CHECK: checking expression 0
// CHECK: types: main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK-EMPTY:
// CHECK: ********** type checking complete **********
// CHECK: fn Print (format_str: String) -> () {
// CHECK: {
// CHECK: return intrinsic_expression(print);
// CHECK-EMPTY:
// CHECK: }
// CHECK-EMPTY:
// CHECK: }
// CHECK: fn main () -> i32 {
// CHECK: {
// CHECK: return 0;
// CHECK-EMPTY:
// CHECK: }
// CHECK-EMPTY:
// CHECK: }
// CHECK: ********** starting execution **********
// CHECK: ********** initializing globals **********
// CHECK: ********** calling main function **********
// CHECK: {
// CHECK: stack:
// CHECK: heap: fun<Print>, fun<main>
// CHECK: }
// CHECK: --- step exp main() (<main()>:0) --->
// CHECK: {
// CHECK: stack: main<0> :: main()<1> :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp main (<main()>:0) --->
// CHECK: {
// CHECK: stack: main()<1>(fun<main>) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp main() (<main()>:0) --->
// CHECK: {
// CHECK: stack: ()<0> :: main()<2>(fun<main>) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp () (<main()>:0) --->
// CHECK: {
// CHECK: stack: main()<2>(fun<main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp main() (<main()>:0) --->
// CHECK: {
// CHECK: stack: {return 0; }<0> :: ScopeAction<0> :: main()<3>(fun<main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step stmt {return 0; } ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:375) --->
// CHECK: {
// CHECK: stack: return 0; <0> :: {return 0; }<1> :: ScopeAction<0> :: main()<3>(fun<main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step stmt return 0;  ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:375) --->
// CHECK: {
// CHECK: stack: return 0;<0> :: return 0; <1> :: {return 0; }<1> :: ScopeAction<0> :: main()<3>(fun<main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step stmt return 0; ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:374) --->
// CHECK: {
// CHECK: stack: 0<0> :: return 0;<1> :: return 0; <1> :: {return 0; }<1> :: ScopeAction<0> :: main()<3>(fun<main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp 0 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:374) --->
// CHECK: {
// CHECK: stack: return 0;<1>(0) :: return 0; <1> :: {return 0; }<1> :: ScopeAction<0> :: main()<3>(fun<main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step stmt return 0; ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:374) --->
// CHECK: {
// CHECK: stack: ScopeAction<0>(0) :: main()<3>(fun<main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: {
// CHECK: stack: main()<3>(fun<main>, (), 0) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp main() (<main()>:0) --->
// CHECK: {
// CHECK: stack: ScopeAction<0>(0)
// CHECK: heap: fun<Print>, fun<main>
// CHECK: values: main: fun<main>, Print: fun<Print>
// CHECK: }
// CHECK: result: 0

package ExecutableSemanticsTest api;

fn main() -> i32 {
  return 0;
}
