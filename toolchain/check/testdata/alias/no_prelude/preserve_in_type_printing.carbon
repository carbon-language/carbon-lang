// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/preserve_in_type_printing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/preserve_in_type_printing.carbon

class C {}
alias A = C;

// The diagnostics below should refer to `A`, not `C`.

// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+7]]:15: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn F() -> A { return; }
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+4]]:8: note: return type of function is `A` [ReturnTypeHereNote]
// CHECK:STDERR: fn F() -> A { return; }
// CHECK:STDERR:        ^~~~
// CHECK:STDERR:
fn F() -> A { return; }
// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+7]]:16: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn G() -> A* { return; }
// CHECK:STDERR:                ^~~~~~~
// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+4]]:8: note: return type of function is `A*` [ReturnTypeHereNote]
// CHECK:STDERR: fn G() -> A* { return; }
// CHECK:STDERR:        ^~~~~
// CHECK:STDERR:
fn G() -> A* { return; }
// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+7]]:22: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn H() -> const A* { return; }
// CHECK:STDERR:                      ^~~~~~~
// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+4]]:8: note: return type of function is `const A*` [ReturnTypeHereNote]
// CHECK:STDERR: fn H() -> const A* { return; }
// CHECK:STDERR:        ^~~~~~~~~~~
// CHECK:STDERR:
fn H() -> const A* { return; }
// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+7]]:26: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn I() -> array(A, 42) { return; }
// CHECK:STDERR:                          ^~~~~~~
// CHECK:STDERR: preserve_in_type_printing.carbon:[[@LINE+4]]:8: note: return type of function is `[A; 42]` [ReturnTypeHereNote]
// CHECK:STDERR: fn I() -> array(A, 42) { return; }
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn I() -> array(A, 42) { return; }

// CHECK:STDOUT: --- preserve_in_type_printing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_42, %C [concrete]
// CHECK:STDOUT:   %I.type: type = fn_type @I [concrete]
// CHECK:STDOUT:   %I: %I.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %A: type = bind_alias A, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %ptr.019 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.019 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A [concrete = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %A.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     %return.param: ref %ptr.019 = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.019 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %return.patt: %ptr.801 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.801 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A [concrete = constants.%C]
// CHECK:STDOUT:     %const: type = const_type %A.ref [concrete = constants.%const]
// CHECK:STDOUT:     %ptr: type = ptr_type %const [concrete = constants.%ptr.801]
// CHECK:STDOUT:     %return.param: ref %ptr.801 = out_param call_param0
// CHECK:STDOUT:     %return: ref %ptr.801 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [concrete = constants.%I] {
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A [concrete = constants.%C]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %A.ref [concrete = constants.%array_type]
// CHECK:STDOUT:     %return.param: ref %array_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %ptr.019 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> %ptr.801 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I() -> %return.param_patt: %array_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
