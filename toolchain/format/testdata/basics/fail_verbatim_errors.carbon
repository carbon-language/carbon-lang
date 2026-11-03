// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/fail_verbatim_errors.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/fail_verbatim_errors.carbon

// --- fail_missing_semicolon.carbon

fn  F ( )  {
  bar
}

// --- fail_unclosed_brace.carbon

fn  F ( )  {
  var   x : i32 = ( a [ 1 ) + 1 ;

fn  G ( )  { }

// --- fail_verbatim_region.carbon

// Outermost error subtrees keep their original spacing and interior comments,
// while surrounding well-formed code and comments between separate error
// subtrees still reformat.
fn  F ( )  {
  var  ok : i32  =  1 ;
  var broken  :  = 2  +  ;
  var  also_ok : i32  =  3 ;
}

fn  G ( )  {
  var  x : i32  =  2  +  // odd   comment
      ;
}

fn  H ( )  {
  ==  1  +  ;
      // between
  ==  2  +  ;
}

// --- fail_recovery_and_eof_comment.carbon

var class: i32 = ;
fn  AfterRecovery ( )  { }
var incomplete: i32 =
// keep me

// --- AUTOUPDATE-SPLIT

// CHECK:STDERR: fail_missing_semicolon.carbon:4:1: error: expected `;` after expression statement [ExpectedExprSemi]
// CHECK:STDERR:
// CHECK:STDERR: fail_unclosed_brace.carbon:2:12: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_unclosed_brace.carbon:4:1: note: possibly missing `}` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_unclosed_brace.carbon:3:23: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
// CHECK:STDERR: fail_unclosed_brace.carbon:3:26: note: possibly missing `]` here [PossiblyMissingBracketHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:7:18: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:7:26: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:13:7: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:17:3-4: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:17:13: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:19:3-4: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_verbatim_region.carbon:19:13: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_recovery_and_eof_comment.carbon:2:10: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR:
// CHECK:STDERR: fail_recovery_and_eof_comment.carbon:2:18: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_recovery_and_eof_comment.carbon:6:1: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
// CHECK:STDERR: fail_recovery_and_eof_comment.carbon:6:1: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR:
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   bar
// CHECK:STDOUT: }
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var x: i32 = (a[1) + 1;
// CHECK:STDOUT:
// CHECK:STDOUT: fn G() {}
// CHECK:STDOUT: // Outermost error subtrees keep their original spacing and interior comments,
// CHECK:STDOUT: // while surrounding well-formed code and comments between separate error
// CHECK:STDOUT: // subtrees still reformat.
// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var ok: i32 = 1;
// CHECK:STDOUT:   var broken  :  = 2  +  ;
// CHECK:STDOUT:   var also_ok: i32 = 3;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn G() {
// CHECK:STDOUT:   var x: i32 = 2  +  // odd   comment
// CHECK:STDOUT:       ;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn H() {
// CHECK:STDOUT:   ==  1  +  ;
// CHECK:STDOUT:   // between
// CHECK:STDOUT:   ==  2  +  ;
// CHECK:STDOUT: }
// CHECK:STDOUT: var class: i32 = ;
// CHECK:STDOUT: fn AfterRecovery() {}
// CHECK:STDOUT: var incomplete: i32 =
// CHECK:STDOUT: // keep me
