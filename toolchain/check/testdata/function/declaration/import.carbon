// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/import.carbon

// ============================================================================
// Setup files
// ============================================================================

// --- api.carbon

library "[[@TEST_NAME]]";

fn A();
fn B(b: i32) -> i32;
fn C(c: (i32,)) -> {.c: i32};
extern fn D();

namespace NS;
fn NS.E();

// --- extern_api.carbon

library "[[@TEST_NAME]]";

extern library "redecl_extern_api" fn A();
extern library "redecl_extern_api" fn B(b: i32) -> i32;
extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
extern library "redecl_extern_api" fn D();

namespace NS;
extern library "redecl_extern_api" fn NS.E();

// ============================================================================
// Test files
// ============================================================================

// --- basics.carbon

library "[[@TEST_NAME]]";

import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- fail_redecl_api.carbon

library "[[@TEST_NAME]]";

import library "api";

// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclarations of `fn A` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
extern fn A();
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclarations of `fn B` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-14]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:5:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn B(b: i32) -> i32;
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclarations of `fn C` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-23]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:6:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclaration of `fn D` is redundant [RedeclRedundant]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-32]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:7:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn D();
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclarations of `fn E` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-41]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:10:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
extern fn NS.E();

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- redecl_extern_api.carbon

library "[[@TEST_NAME]]";

import library "extern_api";

extern fn A();
extern fn B(b: i32) -> i32;
extern fn C(c: (i32,)) -> {.c: i32};
extern fn D();
extern fn NS.E();

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- fail_merge.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_merge.carbon:[[@LINE+45]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:4:1: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+41]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+36]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:5:1: error: duplicate name `B` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+32]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+27]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:6:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+23]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:6:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+18]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:7:1: error: duplicate name `D` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:7:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:10:1: error: duplicate name `E` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:10:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR:
import library "api";
import library "extern_api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- fail_merge_reverse.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+45]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:4:1: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+41]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+36]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:5:1: error: duplicate name `B` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+32]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+27]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:6:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+23]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:6:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+18]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:7:1: error: duplicate name `D` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:7:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:10:1: error: duplicate name `E` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:10:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "extern_api";
import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- unloaded.carbon

library "[[@TEST_NAME]]";

import library "api";

// --- unloaded_extern.carbon

library "[[@TEST_NAME]]";

import library "extern_api";

// CHECK:STDOUT: --- api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %i32.loc5_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %c.patt: %tuple.type.a1c = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.a1c = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:     %return.patt: %struct_type.c = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.c = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.a1c = value_param call_param0
// CHECK:STDOUT:     %.loc6_14.1: type = splice_block %.loc6_14.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc6_14.2: %tuple.type.85c = tuple_literal (%i32.loc6_10)
// CHECK:STDOUT:       %.loc6_14.3: type = converted %.loc6_14.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.a1c = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.a1c) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_52: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5_52: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %i32.loc5_44 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5_44: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5_44: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %c.patt: %tuple.type.a1c = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.a1c = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:     %return.patt: %struct_type.c = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.c = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_60: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_60: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.a1c = value_param call_param0
// CHECK:STDOUT:     %.loc6_49.1: type = splice_block %.loc6_49.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:       %int_32.loc6_45: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_45: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc6_49.2: %tuple.type.85c = tuple_literal (%i32.loc6_45)
// CHECK:STDOUT:       %.loc6_49.3: type = converted %.loc6_49.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.a1c = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c.param_patt: %tuple.type.a1c) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A: %A.type = import_ref Main//api, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: %B.type = import_ref Main//api, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//api, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: %D.type = import_ref Main//api, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = %Main.E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.E: %E.type = import_ref Main//api, E, loaded [concrete = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc6_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc6_9.1: type = splice_block %.loc6_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_9.3: type = converted %.loc6_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc7_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc7_8: type = splice_block %i32.loc7 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %struct_type.c = binding_pattern c
// CHECK:STDOUT:     %.loc8_1: %struct_type.c = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %.loc8_16: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %empty_tuple.type = binding_pattern d
// CHECK:STDOUT:     %.loc9_1: %empty_tuple.type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %.loc9_9.1: type = splice_block %.loc9_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc9_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc9_9.3: type = converted %.loc9_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %empty_tuple.type = binding_pattern e
// CHECK:STDOUT:     %.loc10_1: %empty_tuple.type = var_pattern %e.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %.loc10_9.1: type = splice_block %.loc10_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc10_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_9.3: type = converted %.loc10_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32 [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.a1c) -> %struct_type.c [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc7: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc7_16.1: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_16.2: <bound method> = bound_method %int_1.loc7, %specific_fn.loc7 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc7: init %i32 = call %bound_method.loc7_16.2(%int_1.loc7) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_16.1: %i32 = value_of_initializer %int.convert_checked.loc7 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_16.2: %i32 = converted %int_1.loc7, %.loc7_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc7_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc8_25.1: %tuple.type.985 = tuple_literal (%int_1.loc8)
// CHECK:STDOUT:   %impl.elem0.loc8: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc8_25.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_25.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %bound_method.loc8_25.2(%int_1.loc8) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = value_of_initializer %int.convert_checked.loc8 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_25.3: %i32 = converted %int_1.loc8, %.loc8_25.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc8_25.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc8_25.4: %tuple.type.a1c = converted %.loc8_25.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc8_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%Main.E [concrete = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = file.%E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {
// CHECK:STDOUT:     %int_32.loc23_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %i32.loc23_16 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc23_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc23_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {} {
// CHECK:STDOUT:     %int_32.loc32_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc32_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.a1c = value_param call_param0
// CHECK:STDOUT:     %.loc32_21.1: type = splice_block %.loc32_21.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:       %int_32.loc32_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc32_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc32_21.2: %tuple.type.85c = tuple_literal (%i32.loc32_17)
// CHECK:STDOUT:       %.loc32_21.3: type = converted %.loc32_21.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.a1c = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc52_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc52_9.1: type = splice_block %.loc52_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc52_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc52_9.3: type = converted %.loc52_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc53_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc53_8: type = splice_block %i32.loc53 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc53: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc53: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %struct_type.c = binding_pattern c
// CHECK:STDOUT:     %.loc54_1: %struct_type.c = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %.loc54_16: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %int_32.loc54: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc54: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %empty_tuple.type = binding_pattern d
// CHECK:STDOUT:     %.loc55_1: %empty_tuple.type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %.loc55_9.1: type = splice_block %.loc55_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc55_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc55_9.3: type = converted %.loc55_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %empty_tuple.type = binding_pattern e
// CHECK:STDOUT:     %.loc56_1: %empty_tuple.type = var_pattern %e.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %.loc56_9.1: type = splice_block %.loc56_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc56_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc56_9.3: type = converted %.loc56_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32 [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.a1c) -> %struct_type.c [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc53: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc53_16.1: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc53: <specific function> = specific_function %impl.elem0.loc53, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_16.2: <bound method> = bound_method %int_1.loc53, %specific_fn.loc53 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc53: init %i32 = call %bound_method.loc53_16.2(%int_1.loc53) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_16.1: %i32 = value_of_initializer %int.convert_checked.loc53 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_16.2: %i32 = converted %int_1.loc53, %.loc53_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc53_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc54: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc54_25.1: %tuple.type.985 = tuple_literal (%int_1.loc54)
// CHECK:STDOUT:   %impl.elem0.loc54: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc54_25.1: <bound method> = bound_method %int_1.loc54, %impl.elem0.loc54 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc54: <specific function> = specific_function %impl.elem0.loc54, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc54_25.2: <bound method> = bound_method %int_1.loc54, %specific_fn.loc54 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc54: init %i32 = call %bound_method.loc54_25.2(%int_1.loc54) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc54_25.2: %i32 = value_of_initializer %int.convert_checked.loc54 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc54_25.3: %i32 = converted %int_1.loc54, %.loc54_25.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc54_25.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc54_25.4: %tuple.type.a1c = converted %.loc54_25.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc54_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9ba: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.6da: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.b97: <witness> = impl_witness (imports.%Core.import_ref.a86), @impl.c81(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.ed5: type = fn_type @Convert.2, @impl.c81(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.16d: %Convert.type.ed5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.9ba = facet_value Core.IntLiteral, (%impl_witness.b97) [concrete]
// CHECK:STDOUT:   %.b14: type = fn_type_with_self_type %Convert.type.6da, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.16d [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.16d, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.dd4 = tuple_value (%int_1.47b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//extern_api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = file.%E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {
// CHECK:STDOUT:     %int_32.loc7_24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc7: type = splice_block %i32.loc7_16 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc7_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc7_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {} {
// CHECK:STDOUT:     %int_32.loc8_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.dd4 = value_param call_param0
// CHECK:STDOUT:     %.loc8_21.1: type = splice_block %.loc8_21.3 [concrete = constants.%tuple.type.dd4] {
// CHECK:STDOUT:       %int_32.loc8_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc8_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc8_21.2: %tuple.type.85c = tuple_literal (%i32.loc8_17)
// CHECK:STDOUT:       %.loc8_21.3: type = converted %.loc8_21.2, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.dd4 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc12_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc12_9.1: type = splice_block %.loc12_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_9.3: type = converted %.loc12_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc13_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc13_8: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %struct_type.c = binding_pattern c
// CHECK:STDOUT:     %.loc14_1: %struct_type.c = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %.loc14_16: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %empty_tuple.type = binding_pattern d
// CHECK:STDOUT:     %.loc15_1: %empty_tuple.type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %.loc15_9.1: type = splice_block %.loc15_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc15_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_9.3: type = converted %.loc15_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %empty_tuple.type = binding_pattern e
// CHECK:STDOUT:     %.loc16_1: %empty_tuple.type = var_pattern %e.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %.loc16_9.1: type = splice_block %.loc16_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc16_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_9.3: type = converted %.loc16_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c.param_patt: %tuple.type.dd4) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc13: %.b14 = impl_witness_access constants.%impl_witness.b97, element0 [concrete = constants.%Convert.16d]
// CHECK:STDOUT:   %bound_method.loc13_16.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_16.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %bound_method.loc13_16.2(%int_1.loc13) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc13_16.1: %i32 = value_of_initializer %int.convert_checked.loc13 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc13_16.2: %i32 = converted %int_1.loc13, %.loc13_16.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc13_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc14_25.1: %tuple.type.985 = tuple_literal (%int_1.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14: %.b14 = impl_witness_access constants.%impl_witness.b97, element0 [concrete = constants.%Convert.16d]
// CHECK:STDOUT:   %bound_method.loc14_25.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_25.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %bound_method.loc14_25.2(%int_1.loc14) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc14_25.2: %i32 = value_of_initializer %int.convert_checked.loc14 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc14_25.3: %i32 = converted %int_1.loc14, %.loc14_25.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %tuple: %tuple.type.dd4 = tuple_value (%.loc14_25.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_25.4: %tuple.type.dd4 = converted %.loc14_25.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc14_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A: %A.type = import_ref Main//api, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: %B.type = import_ref Main//api, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//api, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: %D.type = import_ref Main//api, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = %Main.E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.E: %E.type = import_ref Main//api, E, loaded [concrete = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc52_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc52_9.1: type = splice_block %.loc52_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc52_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc52_9.3: type = converted %.loc52_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc53_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc53_8: type = splice_block %i32.loc53 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc53: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc53: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %struct_type.c = binding_pattern c
// CHECK:STDOUT:     %.loc54_1: %struct_type.c = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %.loc54_16: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %int_32.loc54: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc54: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %empty_tuple.type = binding_pattern d
// CHECK:STDOUT:     %.loc55_1: %empty_tuple.type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %.loc55_9.1: type = splice_block %.loc55_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc55_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc55_9.3: type = converted %.loc55_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %empty_tuple.type = binding_pattern e
// CHECK:STDOUT:     %.loc56_1: %empty_tuple.type = var_pattern %e.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %.loc56_9.1: type = splice_block %.loc56_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc56_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc56_9.3: type = converted %.loc56_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32 [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.a1c) -> %struct_type.c [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E() [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc53: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc53_16.1: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc53: <specific function> = specific_function %impl.elem0.loc53, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_16.2: <bound method> = bound_method %int_1.loc53, %specific_fn.loc53 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc53: init %i32 = call %bound_method.loc53_16.2(%int_1.loc53) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_16.1: %i32 = value_of_initializer %int.convert_checked.loc53 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_16.2: %i32 = converted %int_1.loc53, %.loc53_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc53_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc54: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc54_25.1: %tuple.type.985 = tuple_literal (%int_1.loc54)
// CHECK:STDOUT:   %impl.elem0.loc54: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc54_25.1: <bound method> = bound_method %int_1.loc54, %impl.elem0.loc54 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc54: <specific function> = specific_function %impl.elem0.loc54, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc54_25.2: <bound method> = bound_method %int_1.loc54, %specific_fn.loc54 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc54: init %i32 = call %bound_method.loc54_25.2(%int_1.loc54) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc54_25.2: %i32 = value_of_initializer %int.convert_checked.loc54 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc54_25.3: %i32 = converted %int_1.loc54, %.loc54_25.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc54_25.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc54_25.4: %tuple.type.a1c = converted %.loc54_25.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc54_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%Main.E [concrete = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_merge_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A: %A.type = import_ref Main//extern_api, A, loaded [concrete = constants.%A]
// CHECK:STDOUT:   %Main.B: %B.type = import_ref Main//extern_api, B, loaded [concrete = constants.%B]
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//extern_api, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.D: %D.type = import_ref Main//extern_api, D, loaded [concrete = constants.%D]
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//extern_api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = %Main.E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.E: %E.type = import_ref Main//extern_api, E, loaded [concrete = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc52_1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc52_9.1: type = splice_block %.loc52_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc52_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc52_9.3: type = converted %.loc52_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc53_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc53_8: type = splice_block %i32.loc53 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc53: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc53: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %struct_type.c = binding_pattern c
// CHECK:STDOUT:     %.loc54_1: %struct_type.c = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %.loc54_16: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %int_32.loc54: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc54: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %empty_tuple.type = binding_pattern d
// CHECK:STDOUT:     %.loc55_1: %empty_tuple.type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %.loc55_9.1: type = splice_block %.loc55_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc55_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc55_9.3: type = converted %.loc55_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %empty_tuple.type = binding_pattern e
// CHECK:STDOUT:     %.loc56_1: %empty_tuple.type = var_pattern %e.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %.loc56_9.1: type = splice_block %.loc56_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc56_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc56_9.3: type = converted %.loc56_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c.param_patt: %tuple.type.a1c) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc53: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc53_16.1: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc53: <specific function> = specific_function %impl.elem0.loc53, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc53_16.2: <bound method> = bound_method %int_1.loc53, %specific_fn.loc53 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc53: init %i32 = call %bound_method.loc53_16.2(%int_1.loc53) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_16.1: %i32 = value_of_initializer %int.convert_checked.loc53 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc53_16.2: %i32 = converted %int_1.loc53, %.loc53_16.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc53_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc54: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc54_25.1: %tuple.type.985 = tuple_literal (%int_1.loc54)
// CHECK:STDOUT:   %impl.elem0.loc54: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc54_25.1: <bound method> = bound_method %int_1.loc54, %impl.elem0.loc54 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc54: <specific function> = specific_function %impl.elem0.loc54, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc54_25.2: <bound method> = bound_method %int_1.loc54, %specific_fn.loc54 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc54: init %i32 = call %bound_method.loc54_25.2(%int_1.loc54) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc54_25.2: %i32 = value_of_initializer %int.convert_checked.loc54 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc54_25.3: %i32 = converted %int_1.loc54, %.loc54_25.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc54_25.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc54_25.4: %tuple.type.a1c = converted %.loc54_25.1, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc54_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%Main.D [concrete = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [concrete = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%Main.E [concrete = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A = import_ref Main//api, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//api, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//api, C, unloaded
// CHECK:STDOUT:   %Main.D = import_ref Main//api, D, unloaded
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = %Main.E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unloaded_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.A = import_ref Main//extern_api, A, unloaded
// CHECK:STDOUT:   %Main.B = import_ref Main//extern_api, B, unloaded
// CHECK:STDOUT:   %Main.C = import_ref Main//extern_api, C, unloaded
// CHECK:STDOUT:   %Main.D = import_ref Main//extern_api, D, unloaded
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//extern_api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = %Main.E
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%Main.A
// CHECK:STDOUT:     .B = imports.%Main.B
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .D = imports.%Main.D
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
