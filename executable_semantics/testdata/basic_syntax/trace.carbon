// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: executable_semantics --trace --trace_bison=false %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
//
// The below output should only be used for basic validation -- changes are
// okay.
//
// AUTOUPDATE: executable_semantics --trace --trace_bison=false %s
// CHECK: ********** source program **********
// CHECK: fn Print (format_str: String) -> () {
// CHECK: {
// CHECK: return intrinsic_expression(print);
// CHECK: }
// CHECK-EMPTY:
// CHECK: }
// CHECK: fn Main () -> i32 {
// CHECK: {
// CHECK: return 0;
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
// CHECK: --- step pattern () ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:190) --->
// CHECK: --- step pattern i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:190) --->
// CHECK: --- step exp i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:190) --->
// CHECK: checking pattern (format_str: String)
// CHECK: types: Main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: checking pattern format_str: String
// CHECK: types: Main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: checking pattern String
// CHECK: types: Main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: checking expression String
// CHECK: types: Main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: Main: fun<Main>, Print: fun<Print>
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
// CHECK: types: format_str: String, Main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: checking pattern ()
// CHECK: types: Main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: --- step pattern () ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:190) --->
// CHECK: --- step pattern i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:190) --->
// CHECK: --- step exp i32 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:190) --->
// CHECK: checking expression 0
// CHECK: types: Main: fn () -> i32, Print: fn (String) -> ()
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK-EMPTY:
// CHECK: ********** type checking complete **********
// CHECK: fn Print (format_str: String) -> () {
// CHECK: {
// CHECK: return intrinsic_expression(print);
// CHECK: }
// CHECK-EMPTY:
// CHECK: }
// CHECK: fn Main () -> i32 {
// CHECK: {
// CHECK: return 0;
// CHECK: }
// CHECK-EMPTY:
// CHECK: }
// CHECK: ********** starting execution **********
// CHECK: ********** initializing globals **********
// CHECK: ********** calling main function **********
// CHECK: {
// CHECK: stack:
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: }
// CHECK: --- step exp Main() (<Main()>:0) --->
// CHECK: {
// CHECK: stack: Main<0> :: Main()<1> :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp Main (<Main()>:0) --->
// CHECK: {
// CHECK: stack: Main()<1>(fun<Main>) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp Main() (<Main()>:0) --->
// CHECK: {
// CHECK: stack: ()<0> :: Main()<2>(fun<Main>) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp () (<Main()>:0) --->
// CHECK: {
// CHECK: stack: Main()<2>(fun<Main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp Main() (<Main()>:0) --->
// CHECK: {
// CHECK: stack: {return 0;}<0> :: ScopeAction<0> :: Main()<3>(fun<Main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step stmt {return 0;} ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:192) --->
// CHECK: {
// CHECK: stack: return 0;<0> :: {return 0;}<1> :: ScopeAction<0> :: Main()<3>(fun<Main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step stmt return 0; ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:191) --->
// CHECK: {
// CHECK: stack: 0<0> :: return 0;<1> :: {return 0;}<1> :: ScopeAction<0> :: Main()<3>(fun<Main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp 0 ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:191) --->
// CHECK: {
// CHECK: stack: return 0;<1>(0) :: {return 0;}<1> :: ScopeAction<0> :: Main()<3>(fun<Main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step stmt return 0; ({{.*}}/executable_semantics/testdata/basic_syntax/trace.carbon:191) --->
// CHECK: {
// CHECK: stack: ScopeAction<0>(0) :: Main()<3>(fun<Main>, ()) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: {
// CHECK: stack: Main()<3>(fun<Main>, (), 0) :: ScopeAction<0>
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: --- step exp Main() (<Main()>:0) --->
// CHECK: {
// CHECK: stack: ScopeAction<0>(0)
// CHECK: heap: fun<Print>, fun<Main>
// CHECK: values: Main: fun<Main>, Print: fun<Print>
// CHECK: }
// CHECK: result: 0

package ExecutableSemanticsTest api;

fn Main() -> i32 {
  return 0;
}
