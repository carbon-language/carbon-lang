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
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-7: error: redeclaration of `fn A` is redundant [RedeclRedundant]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-7: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn A();

fn B(x: ());
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-12: error: redeclaration of `fn B` is redundant [RedeclRedundant]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-12: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn B(x: ());

fn C();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-12: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-7: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
// CHECK:STDERR:
fn C(x: ());

fn D() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-7: error: redeclaration of `fn D` is redundant [RedeclRedundant]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-8: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR:
fn D();

fn E() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+3]]:1-8: error: redefinition of `fn E` [RedeclRedef]
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-2]]:1-8: note: previously defined here [RedeclPrevDef]
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
// CHECK:STDOUT:   %x.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.322e3b.1: type = fn_type @C.loc27 [concrete]
// CHECK:STDOUT:   %C.010386.1: %C.type.322e3b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.322e3b.2: type = fn_type @C.loc31 [concrete]
// CHECK:STDOUT:   %C.010386.2: %C.type.322e3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type.53e05b.1: type = fn_type @E.loc39 [concrete]
// CHECK:STDOUT:   %E.3f2512.1: %E.type.53e05b.1 = struct_value () [concrete]
// CHECK:STDOUT:   %E.type.53e05b.2: type = fn_type @E.loc43 [concrete]
// CHECK:STDOUT:   %E.3f2512.2: %E.type.53e05b.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl.loc15
// CHECK:STDOUT:     .B = %B.decl.loc21
// CHECK:STDOUT:     .C = %C.decl.loc27
// CHECK:STDOUT:     .D = %D.decl.loc33
// CHECK:STDOUT:     .E = %E.decl.loc39
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc15: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %A.decl.loc19: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc21: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc21: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc21_10.1: type = splice_block %.loc21_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc21_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc21_10.3: type = converted %.loc21_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc21: %empty_tuple.type = wrapper_binding x, %x.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc25: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc25: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc25_10.1: type = splice_block %.loc25_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc25_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc25_10.3: type = converted %.loc25_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc25: %empty_tuple.type = wrapper_binding x, %x.param.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc27: %C.type.322e3b.1 = fn_decl @C.loc27 [concrete = constants.%C.010386.1] {} {}
// CHECK:STDOUT:   %C.decl.loc31: %C.type.322e3b.2 = fn_decl @C.loc31 [concrete = constants.%C.010386.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc31_10.1: type = splice_block %.loc31_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc31_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc31_10.3: type = converted %.loc31_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc33: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %D.decl.loc37: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl.loc39: %E.type.53e05b.1 = fn_decl @E.loc39 [concrete = constants.%E.3f2512.1] {} {}
// CHECK:STDOUT:   %E.decl.loc43: %E.type.53e05b.2 = fn_decl @E.loc43 [concrete = constants.%E.3f2512.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%x.param.loc21: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc27();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.loc31(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.loc39() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.loc43() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
