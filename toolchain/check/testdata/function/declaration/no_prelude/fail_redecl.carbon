// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_redecl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_redecl.carbon

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
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.de0bfe.1: type = fn_type @C.1 [concrete]
// CHECK:STDOUT:   %C.1b0370.1: %C.type.de0bfe.1 = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.de0bfe.2: type = fn_type @C.2 [concrete]
// CHECK:STDOUT:   %C.1b0370.2: %C.type.de0bfe.2 = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type.851869.1: type = fn_type @E.1 [concrete]
// CHECK:STDOUT:   %E.237d29.1: %E.type.851869.1 = struct_value () [concrete]
// CHECK:STDOUT:   %E.type.851869.2: type = fn_type @E.2 [concrete]
// CHECK:STDOUT:   %E.237d29.2: %E.type.851869.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl.loc11
// CHECK:STDOUT:     .B = %B.decl.loc21
// CHECK:STDOUT:     .C = %C.decl.loc31
// CHECK:STDOUT:     .D = %D.decl.loc41
// CHECK:STDOUT:     .E = %E.decl.loc51
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc11: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %A.decl.loc19: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl.loc21: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc21: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc21_10.1: type = splice_block %.loc21_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc21_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc21_10.3: type = converted %.loc21_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc21: %empty_tuple.type = bind_name x, %x.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc29: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param.loc29: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc29_10.1: type = splice_block %.loc29_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc29_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc29_10.3: type = converted %.loc29_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.loc29: %empty_tuple.type = bind_name x, %x.param.loc29
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc31: %C.type.de0bfe.1 = fn_decl @C.1 [concrete = constants.%C.1b0370.1] {} {}
// CHECK:STDOUT:   %C.decl.loc39: %C.type.de0bfe.2 = fn_decl @C.2 [concrete = constants.%C.1b0370.2] {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: %empty_tuple.type = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc39_10.1: type = splice_block %.loc39_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc39_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc39_10.3: type = converted %.loc39_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc41: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %D.decl.loc49: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl.loc51: %E.type.851869.1 = fn_decl @E.1 [concrete = constants.%E.237d29.1] {} {}
// CHECK:STDOUT:   %E.decl.loc59: %E.type.851869.2 = fn_decl @E.2 [concrete = constants.%E.237d29.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%x.param_patt: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.2(%x.param_patt: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
