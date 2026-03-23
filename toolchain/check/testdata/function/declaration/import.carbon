// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDERR: api.carbon:23:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
extern fn A();
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclarations of `fn B` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-14]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:24:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn B(b: i32) -> i32;
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclarations of `fn C` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-23]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:25:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclaration of `fn D` is redundant [RedeclRedundant]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-32]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:26:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn D();
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+8]]:1: error: redeclarations of `fn E` must match use of `extern` [RedeclExternMismatch]
// CHECK:STDERR: extern fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-41]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:29:1: note: previously declared here [RedeclPrevDecl]
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
// CHECK:STDERR: extern_api.carbon:35:1: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+41]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:23:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+36]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:36:1: error: duplicate name `B` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+32]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:24:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+27]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:37:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+23]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:25:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+18]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:38:1: error: duplicate name `D` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:26:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:41:1: error: duplicate name `E` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:29:1: note: name is previously declared here [NameDeclPrevious]
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
// CHECK:STDERR: api.carbon:23:1: error: duplicate name `A` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+41]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:35:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+36]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:24:1: error: duplicate name `B` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+32]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:36:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+27]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:25:1: error: duplicate name `C` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+23]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:37:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+18]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:26:1: error: duplicate name `D` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:38:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:29:1: error: duplicate name `E` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:41:1: note: name is previously declared here [NameDeclPrevious]
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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.b74: type = pattern_type %tuple.type.a1c [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %.da9: Core.Form = init_form %struct_type.c [concrete]
// CHECK:STDOUT:   %pattern_type.688: type = pattern_type %struct_type.c [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc17 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc17: Core.Form = init_form %i32.loc17 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.b74 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.b74 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.688 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.688 = return_slot_pattern %return.param_patt, %struct_type.c [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %.loc35: Core.Form = init_form %struct_type.c [concrete = constants.%.da9]
// CHECK:STDOUT:     %c.param: %tuple.type.a1c = value_param call_param0
// CHECK:STDOUT:     %.loc27.1: type = splice_block %.loc27.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:       %i32.loc25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc27.2: %tuple.type.85c = tuple_literal (%i32.loc25) [concrete = constants.%tuple]
// CHECK:STDOUT:       %.loc27.3: type = converted %.loc27.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.a1c = value_binding c, %c.param
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
// CHECK:STDOUT: fn @B(%b.param: %i32) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param: %tuple.type.a1c) -> out %return.param: %struct_type.c;
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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.b74: type = pattern_type %tuple.type.a1c [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %.da9: Core.Form = init_form %struct_type.c [concrete]
// CHECK:STDOUT:   %pattern_type.688: type = pattern_type %struct_type.c [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc23 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %i32.loc23 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.b74 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.b74 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.688 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.688 = return_slot_pattern %return.param_patt, %struct_type.c [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc43: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %.loc44: Core.Form = init_form %struct_type.c [concrete = constants.%.da9]
// CHECK:STDOUT:     %c.param: %tuple.type.a1c = value_param call_param0
// CHECK:STDOUT:     %.loc36.1: type = splice_block %.loc36.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:       %i32.loc34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc36.2: %tuple.type.85c = tuple_literal (%i32.loc34) [concrete = constants.%tuple]
// CHECK:STDOUT:       %.loc36.3: type = converted %.loc36.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.a1c = value_binding c, %c.param
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
// CHECK:STDOUT: extern fn @B(%b.param: %i32) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c.param: %tuple.type.a1c) -> out %return.param: %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.688: type = pattern_type %struct_type.c [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %tuple.246: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc12.1: type = splice_block %.loc12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc12.3: type = converted %.loc12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.688 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.688 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc36: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %i32.loc35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc50.1: type = splice_block %.loc50.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc50.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc50.3: type = converted %.loc50.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.cb1 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.cb1 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc60.1: type = splice_block %.loc60.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc60.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc60.3: type = converted %.loc60.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = ref_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc25: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc25.1: <bound method> = bound_method %int_1.loc25, %impl.elem0.loc25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25.2: <bound method> = bound_method %int_1.loc25, %specific_fn.loc25 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %i32 = call %bound_method.loc25.2(%int_1.loc25) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc25.2: %i32 = converted %int_1.loc25, %.loc25.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc25.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc41: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc43.1: %tuple.type.985 = tuple_literal (%int_1.loc41) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %impl.elem0.loc43: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc43.1: <bound method> = bound_method %int_1.loc41, %impl.elem0.loc43 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc43: <specific function> = specific_function %impl.elem0.loc43, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc43.2: <bound method> = bound_method %int_1.loc41, %specific_fn.loc43 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43: init %i32 = call %bound_method.loc43.2(%int_1.loc41) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc43.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc43.3: %i32 = converted %int_1.loc41, %.loc43.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc43.3) [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %.loc43.4: %tuple.type.a1c = converted %.loc43.1, %tuple [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc43.4)
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
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.896: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %.da9: Core.Form = init_form %struct_type.c [concrete]
// CHECK:STDOUT:   %pattern_type.688: type = pattern_type %struct_type.c [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %tuple.246: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %i32.loc23 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {} {
// CHECK:STDOUT:     %i32.loc41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %.loc42: Core.Form = init_form %struct_type.c [concrete = constants.%.da9]
// CHECK:STDOUT:     %c.param: %tuple.type.a1c = value_param call_param0
// CHECK:STDOUT:     %.loc34.1: type = splice_block %.loc34.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:       %i32.loc32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc34.2: %tuple.type.85c = tuple_literal (%i32.loc32) [concrete = constants.%tuple.896]
// CHECK:STDOUT:       %.loc34.3: type = converted %.loc34.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.a1c = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc62.1: type = splice_block %.loc62.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc62.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc62.3: type = converted %.loc62.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc71: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.688 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.688 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc86: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %i32.loc85: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc100.1: type = splice_block %.loc100.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc100.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc100.3: type = converted %.loc100.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.cb1 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.cb1 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc110.1: type = splice_block %.loc110.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc110.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc110.3: type = converted %.loc110.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = ref_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc75: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc75: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc75.1: <bound method> = bound_method %int_1.loc75, %impl.elem0.loc75 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc75: <specific function> = specific_function %impl.elem0.loc75, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc75.2: <bound method> = bound_method %int_1.loc75, %specific_fn.loc75 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc75: init %i32 = call %bound_method.loc75.2(%int_1.loc75) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc75.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc75 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc75.2: %i32 = converted %int_1.loc75, %.loc75.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc75.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc91: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc93.1: %tuple.type.985 = tuple_literal (%int_1.loc91) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %impl.elem0.loc93: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc93.1: <bound method> = bound_method %int_1.loc91, %impl.elem0.loc93 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc93: <specific function> = specific_function %impl.elem0.loc93, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc93.2: <bound method> = bound_method %int_1.loc91, %specific_fn.loc93 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc93: init %i32 = call %bound_method.loc93.2(%int_1.loc91) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc93.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc93 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc93.3: %i32 = converted %int_1.loc91, %.loc93.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc93.3) [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %.loc93.4: %tuple.type.a1c = converted %.loc93.1, %tuple [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc93.4)
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
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.501: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.4ca: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.f5f: type = pattern_type %struct_type.c [concrete]
// CHECK:STDOUT:   %tuple.type.dd4: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %.dce: Core.Form = init_form %struct_type.c [concrete]
// CHECK:STDOUT:   %D.type: type = fn_type @D [concrete]
// CHECK:STDOUT:   %D: %D.type = struct_value () [concrete]
// CHECK:STDOUT:   %E.type: type = fn_type @E [concrete]
// CHECK:STDOUT:   %E: %E.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.38e: %tuple.type.85c = tuple_value (%i32) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cf3: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.255: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.58d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e45, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.199 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cf3 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.58d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.979: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.7c3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.979, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.47b: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %tuple.7dd: %tuple.type.dd4 = tuple_value (%int_1.47b) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.NS: <namespace> = import_ref Main//extern_api, NS, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %Main.NS, [concrete] {
// CHECK:STDOUT:     .E = file.%E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b25: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.004) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.255)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e45 = impl_witness_table (%Core.import_ref.b25), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core.ece
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
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %i32.loc23 [concrete = constants.%.4ca]
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [concrete = constants.%C] {} {
// CHECK:STDOUT:     %i32.loc41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:     %.loc42: Core.Form = init_form %struct_type.c [concrete = constants.%.dce]
// CHECK:STDOUT:     %c.param: %tuple.type.dd4 = value_param call_param0
// CHECK:STDOUT:     %.loc34.1: type = splice_block %.loc34.3 [concrete = constants.%tuple.type.dd4] {
// CHECK:STDOUT:       %i32.loc32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc34.2: %tuple.type.85c = tuple_literal (%i32.loc32) [concrete = constants.%tuple.38e]
// CHECK:STDOUT:       %.loc34.3: type = converted %.loc34.2, constants.%tuple.type.dd4 [concrete = constants.%tuple.type.dd4]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %tuple.type.dd4 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param call_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc62.1: type = splice_block %.loc62.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc62.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc62.3: type = converted %.loc62.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.501 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.501 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc71: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f5f = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.f5f = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc86: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %i32.loc85: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc100.1: type = splice_block %.loc100.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc100.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc100.3: type = converted %.loc100.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.cb1 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.cb1 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc110.1: type = splice_block %.loc110.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc110.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc110.3: type = converted %.loc110.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = ref_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc75: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc75: %.7c3 = impl_witness_access constants.%ImplicitAs.impl_witness.58d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4]
// CHECK:STDOUT:   %bound_method.loc75.1: <bound method> = bound_method %int_1.loc75, %impl.elem0.loc75 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc75: <specific function> = specific_function %impl.elem0.loc75, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc75.2: <bound method> = bound_method %int_1.loc75, %specific_fn.loc75 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc75: init %i32 = call %bound_method.loc75.2(%int_1.loc75) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc75.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc75 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc75.2: %i32 = converted %int_1.loc75, %.loc75.1 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc75.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc91: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc93.1: %tuple.type.985 = tuple_literal (%int_1.loc91) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %impl.elem0.loc93: %.7c3 = impl_witness_access constants.%ImplicitAs.impl_witness.58d, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.5f4]
// CHECK:STDOUT:   %bound_method.loc93.1: <bound method> = bound_method %int_1.loc91, %impl.elem0.loc93 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc93: <specific function> = specific_function %impl.elem0.loc93, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc93.2: <bound method> = bound_method %int_1.loc91, %specific_fn.loc93 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc93: init %i32 = call %bound_method.loc93.2(%int_1.loc91) [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc93.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc93 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %.loc93.3: %i32 = converted %int_1.loc91, %.loc93.2 [concrete = constants.%int_1.47b]
// CHECK:STDOUT:   %tuple: %tuple.type.dd4 = tuple_value (%.loc93.3) [concrete = constants.%tuple.7dd]
// CHECK:STDOUT:   %.loc93.4: %tuple.type.dd4 = converted %.loc93.1, %tuple [concrete = constants.%tuple.7dd]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc93.4)
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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.688: type = pattern_type %struct_type.c [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %tuple.246: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc16.1: type = splice_block %.loc16.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc16.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16.3: type = converted %.loc16.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.688 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.688 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc40: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %i32.loc39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc54.1: type = splice_block %.loc54.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc54.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc54.3: type = converted %.loc54.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.cb1 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.cb1 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc64.1: type = splice_block %.loc64.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc64.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc64.3: type = converted %.loc64.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = ref_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E [from "api.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc29: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc29: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc29.1: <bound method> = bound_method %int_1.loc29, %impl.elem0.loc29 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29.2: <bound method> = bound_method %int_1.loc29, %specific_fn.loc29 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29: init %i32 = call %bound_method.loc29.2(%int_1.loc29) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc29.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc29.2: %i32 = converted %int_1.loc29, %.loc29.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc29.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc45: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc47.1: %tuple.type.985 = tuple_literal (%int_1.loc45) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %impl.elem0.loc47: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %int_1.loc45, %impl.elem0.loc47 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %int_1.loc45, %specific_fn.loc47 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47: init %i32 = call %bound_method.loc47.2(%int_1.loc45) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc47.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc47.3: %i32 = converted %int_1.loc45, %.loc47.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc47.3) [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %.loc47.4: %tuple.type.a1c = converted %.loc47.1, %tuple [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc47.4)
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
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.688: type = pattern_type %struct_type.c [concrete]
// CHECK:STDOUT:   %C.type: type = fn_type @C [concrete]
// CHECK:STDOUT:   %C: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.378: %tuple.type.985 = tuple_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %tuple.246: %tuple.type.a1c = tuple_value (%int_1.5d2) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc16.1: type = splice_block %.loc16.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc16.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc16.3: type = converted %.loc16.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.688 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.688 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc40: type = splice_block %struct_type.c [concrete = constants.%struct_type.c] {
// CHECK:STDOUT:     %i32.loc39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [concrete = constants.%struct_type.c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %struct_type.c = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc54.1: type = splice_block %.loc54.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc54.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc54.3: type = converted %.loc54.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %empty_tuple.type = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.cb1 = ref_binding_pattern e [concrete]
// CHECK:STDOUT:     %e.var_patt: %pattern_type.cb1 = var_pattern %e.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var %e.var_patt [concrete]
// CHECK:STDOUT:   %.loc64.1: type = splice_block %.loc64.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc64.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc64.3: type = converted %.loc64.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: ref %empty_tuple.type = ref_binding e, %e.var [concrete = %e.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%Main.A [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%Main.B [concrete = constants.%B]
// CHECK:STDOUT:   %int_1.loc29: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc29: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc29.1: <bound method> = bound_method %int_1.loc29, %impl.elem0.loc29 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc29: <specific function> = specific_function %impl.elem0.loc29, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29.2: <bound method> = bound_method %int_1.loc29, %specific_fn.loc29 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29: init %i32 = call %bound_method.loc29.2(%int_1.loc29) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc29.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc29 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc29.2: %i32 = converted %int_1.loc29, %.loc29.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc29.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:   %int_1.loc45: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc47.1: %tuple.type.985 = tuple_literal (%int_1.loc45) [concrete = constants.%tuple.378]
// CHECK:STDOUT:   %impl.elem0.loc47: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc47.1: <bound method> = bound_method %int_1.loc45, %impl.elem0.loc47 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc47: <specific function> = specific_function %impl.elem0.loc47, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc47.2: <bound method> = bound_method %int_1.loc45, %specific_fn.loc47 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47: init %i32 = call %bound_method.loc47.2(%int_1.loc45) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc47.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc47 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc47.3: %i32 = converted %int_1.loc45, %.loc47.2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%.loc47.3) [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %.loc47.4: %tuple.type.a1c = converted %.loc47.1, %tuple [concrete = constants.%tuple.246]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc47.4)
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
// CHECK:STDOUT:   %Main.E = import_ref Main//api, E, unloaded
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
// CHECK:STDOUT:   %Main.E = import_ref Main//extern_api, E, unloaded
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
