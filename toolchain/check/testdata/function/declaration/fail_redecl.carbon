// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_redecl.carbon

fn A();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: redeclaration of `fn A` is redundant [RedeclRedundant]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn A();

fn B(x: ());
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: redeclaration of `fn B` is redundant [RedeclRedundant]
// CHECK:STDERR: fn B(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn B(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn B(x: ());

fn C();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
// CHECK:STDERR: fn C(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
// CHECK:STDERR: fn C();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn C(x: ());

fn D() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: redeclaration of `fn D` is redundant [RedeclRedundant]
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn D() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn D();

fn E() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: redefinition of `fn E` [RedeclRedef]
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn E() {}

// CHECK:STDOUT: --- fail_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.de0bfe.1: type = fn_type @C.loc35 [concrete]
// CHECK:STDOUT:   %C.1b0370.1: %C.type.de0bfe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.de0bfe.2: type = fn_type @C.loc43 [concrete]
// CHECK:STDOUT:   %C.1b0370.2: %C.type.de0bfe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type.851869.1: type = fn_type @E.loc55 [concrete]
// CHECK:STDOUT:   %E.237d29.1: %E.type.851869.1 = struct_value () [concrete]
// CHECK:STDOUT:   %E.type.851869.2: type = fn_type @E.loc63 [concrete]
// CHECK:STDOUT:   %E.237d29.2: %E.type.851869.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl.loc15
// CHECK:STDOUT:     .B = %B.decl.loc25
// CHECK:STDOUT:     .C = %C.decl.loc35
// CHECK:STDOUT:     .D = %D.decl.loc45
// CHECK:STDOUT:     .E = %E.decl.loc55
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc15: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %A.decl.loc23: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc25: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc25: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc25_10.1: type = splice_block %.loc25_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc25_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc25_10.3: type = converted %.loc25_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc25: %empty_tuple.type = value_binding x, %x.param.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc33: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc33: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc33_10.1: type = splice_block %.loc33_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc33_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc33_10.3: type = converted %.loc33_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc33: %empty_tuple.type = value_binding x, %x.param.loc33
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc35: %C.type.de0bfe.1 = fn_decl @C.loc35 [concrete = constants.%C.1b0370.1] {} {}
// CHECK:STDOUT:   %C.decl.loc43: %C.type.de0bfe.2 = fn_decl @C.loc43 [concrete = constants.%C.1b0370.2] {
// CHECK:STDOUT:     %x.patt: %pattern_type = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc43_10.1: type = splice_block %.loc43_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc43_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc43_10.3: type = converted %.loc43_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc45: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %D.decl.loc53: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl.loc55: %E.type.851869.1 = fn_decl @E.loc55 [concrete = constants.%E.237d29.1] {} {}
// CHECK:STDOUT:   %E.decl.loc63: %E.type.851869.2 = fn_decl @E.loc63 [concrete = constants.%E.237d29.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%x.param.loc25: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc35();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc43(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.loc55() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.loc63() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
