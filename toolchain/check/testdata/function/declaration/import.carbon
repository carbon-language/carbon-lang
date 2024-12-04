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
// CHECK:STDERR: extern_api.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+41]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+36]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:5:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+32]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+27]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:6:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+23]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:6:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+18]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:7:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:7:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:10:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
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

// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+44]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:4:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+40]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+35]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:5:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+31]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:5:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+26]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:6:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+22]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:6:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+17]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:7:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+13]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:7:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: api.carbon:10:1: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+4]]:1: in import [InImport]
// CHECK:STDERR: extern_api.carbon:10:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: extern library "redecl_extern_api" fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_9: init type = call constants.%Int(%int_32.loc5_9) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_9.1: type = value_of_initializer %int.make_type_signed.loc5_9 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_9.2: type = converted %int.make_type_signed.loc5_9, %.loc5_9.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_17: init type = call constants.%Int(%int_32.loc5_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_17.1: type = value_of_initializer %int.make_type_signed.loc5_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_17.2: type = converted %int.make_type_signed.loc5_17, %.loc5_17.1 [template = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %c.patt: %tuple.type.2 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.2 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %struct_type.c = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.c = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_10: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_10: init type = call constants.%Int(%int_32.loc6_10) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc6_10)
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_signed.loc6_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_signed.loc6_10, %.loc6_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_14.4: type = converted %.loc6_14.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %int_32.loc6_25: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_25: init type = call constants.%Int(%int_32.loc6_25) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_25.1: type = value_of_initializer %int.make_type_signed.loc6_25 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_25.2: type = converted %int.make_type_signed.loc6_25, %.loc6_25.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %c: %tuple.type.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.2) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %i32 = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc5_44: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_44: init type = call constants.%Int(%int_32.loc5_44) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_44.1: type = value_of_initializer %int.make_type_signed.loc5_44 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_44.2: type = converted %int.make_type_signed.loc5_44, %.loc5_44.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc5_52: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc5_52: init type = call constants.%Int(%int_32.loc5_52) [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_52.1: type = value_of_initializer %int.make_type_signed.loc5_52 [template = constants.%i32]
// CHECK:STDOUT:     %.loc5_52.2: type = converted %int.make_type_signed.loc5_52, %.loc5_52.1 [template = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %c.patt: %tuple.type.2 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %tuple.type.2 = value_param_pattern %c.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %struct_type.c = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %struct_type.c = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_45: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_45: init type = call constants.%Int(%int_32.loc6_45) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_49.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc6_45)
// CHECK:STDOUT:     %.loc6_49.2: type = value_of_initializer %int.make_type_signed.loc6_45 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_49.3: type = converted %int.make_type_signed.loc6_45, %.loc6_49.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_49.4: type = converted %.loc6_49.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %int_32.loc6_60: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc6_60: init type = call constants.%Int(%int_32.loc6_60) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_60.1: type = value_of_initializer %int.make_type_signed.loc6_60 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_60.2: type = converted %int.make_type_signed.loc6_60, %.loc6_60.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %c: %tuple.type.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c.param_patt: %tuple.type.2) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%int_1.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//api, inst+33, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//api, inst+58, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//api, inst+61, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+64, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//api, inst+65, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.7
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:     .B = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = imports.%import_ref.4
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc6_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%int_32.loc7) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_signed.loc7, %.loc7_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc8: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc8: init type = call constants.%Int(%int_32.loc8) [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_signed.loc8 [template = constants.%i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_signed.loc8, %.loc8_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   %.loc9_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_9.2: type = converted %.loc9_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %.loc10_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.1) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc7: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc7: <specific function> = specific_function %Convert.bound.loc7, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc7: init %i32 = call %Convert.specific_fn.loc7(%int_1.loc7) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_16.1: %i32 = value_of_initializer %int.convert_checked.loc7 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc7_16.2: %i32 = converted %int_1.loc7, %.loc7_16.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc7_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc8_25.1: %tuple.type.2 = tuple_literal (%int_1.loc8)
// CHECK:STDOUT:   %impl.elem0.loc8: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc8: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc8: <specific function> = specific_function %Convert.bound.loc8, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc8: init %i32 = call %Convert.specific_fn.loc8(%int_1.loc8) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = value_of_initializer %int.convert_checked.loc8 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc8_25.3: %i32 = converted %int_1.loc8, %.loc8_25.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.loc8_25.3) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_25.4: %tuple.type.1 = converted %.loc8_25.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc8_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_1.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+64, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = file.%E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.7
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {} {
// CHECK:STDOUT:     %int_32.loc23_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc23_16: init type = call constants.%Int(%int_32.loc23_16) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_16.1: type = value_of_initializer %int.make_type_signed.loc23_16 [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_16.2: type = converted %int.make_type_signed.loc23_16, %.loc23_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc23_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc23_24: init type = call constants.%Int(%int_32.loc23_24) [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_24.1: type = value_of_initializer %int.make_type_signed.loc23_24 [template = constants.%i32]
// CHECK:STDOUT:     %.loc23_24.2: type = converted %int.make_type_signed.loc23_24, %.loc23_24.1 [template = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {} {
// CHECK:STDOUT:     %int_32.loc32_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc32_17: init type = call constants.%Int(%int_32.loc32_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc32_21.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc32_17)
// CHECK:STDOUT:     %.loc32_21.2: type = value_of_initializer %int.make_type_signed.loc32_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc32_21.3: type = converted %int.make_type_signed.loc32_17, %.loc32_21.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc32_21.4: type = converted %.loc32_21.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %int_32.loc32_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc32_32: init type = call constants.%Int(%int_32.loc32_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc32_32.1: type = value_of_initializer %int.make_type_signed.loc32_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc32_32.2: type = converted %int.make_type_signed.loc32_32, %.loc32_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %c: %tuple.type.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {} {}
// CHECK:STDOUT:   %.loc52_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc52_9.2: type = converted %.loc52_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc53: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc53: init type = call constants.%Int(%int_32.loc53) [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_8.1: type = value_of_initializer %int.make_type_signed.loc53 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_8.2: type = converted %int.make_type_signed.loc53, %.loc53_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc54: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc54: init type = call constants.%Int(%int_32.loc54) [template = constants.%i32]
// CHECK:STDOUT:   %.loc54_13.1: type = value_of_initializer %int.make_type_signed.loc54 [template = constants.%i32]
// CHECK:STDOUT:   %.loc54_13.2: type = converted %int.make_type_signed.loc54, %.loc54_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   %.loc55_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc55_9.2: type = converted %.loc55_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %.loc56_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc56_9.2: type = converted %.loc56_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.2) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc53: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc53: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc53: <specific function> = specific_function %Convert.bound.loc53, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc53: init %i32 = call %Convert.specific_fn.loc53(%int_1.loc53) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc53_16.1: %i32 = value_of_initializer %int.convert_checked.loc53 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc53_16.2: %i32 = converted %int_1.loc53, %.loc53_16.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc53_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %int_1.loc54: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc54_25.1: %tuple.type.3 = tuple_literal (%int_1.loc54)
// CHECK:STDOUT:   %impl.elem0.loc54: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc54: <bound method> = bound_method %int_1.loc54, %impl.elem0.loc54 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc54: <specific function> = specific_function %Convert.bound.loc54, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc54: init %i32 = call %Convert.specific_fn.loc54(%int_1.loc54) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc54_25.2: %i32 = value_of_initializer %int.convert_checked.loc54 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc54_25.3: %i32 = converted %int_1.loc54, %.loc54_25.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc54_25.3) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc54_25.4: %tuple.type.2 = converted %.loc54_25.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc54_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_1.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//extern_api, inst+64, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = file.%E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.7
// CHECK:STDOUT:     .ImplicitAs = %import_ref.8
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
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
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {} {
// CHECK:STDOUT:     %int_32.loc7_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_16: init type = call constants.%Int(%int_32.loc7_16) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_16.1: type = value_of_initializer %int.make_type_signed.loc7_16 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_16.2: type = converted %int.make_type_signed.loc7_16, %.loc7_16.1 [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc7_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc7_24: init type = call constants.%Int(%int_32.loc7_24) [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_24.1: type = value_of_initializer %int.make_type_signed.loc7_24 [template = constants.%i32]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %int.make_type_signed.loc7_24, %.loc7_24.1 [template = constants.%i32]
// CHECK:STDOUT:     %b.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {} {
// CHECK:STDOUT:     %int_32.loc8_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc8_17: init type = call constants.%Int(%int_32.loc8_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_21.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc8_17)
// CHECK:STDOUT:     %.loc8_21.2: type = value_of_initializer %int.make_type_signed.loc8_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_21.3: type = converted %int.make_type_signed.loc8_17, %.loc8_21.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_21.4: type = converted %.loc8_21.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %int_32.loc8_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc8_32: init type = call constants.%Int(%int_32.loc8_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_32.1: type = value_of_initializer %int.make_type_signed.loc8_32 [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %int.make_type_signed.loc8_32, %.loc8_32.1 [template = constants.%i32]
// CHECK:STDOUT:     %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:     %c.param: %tuple.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %c: %tuple.type.2 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %struct_type.c = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %struct_type.c = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {} {}
// CHECK:STDOUT:   %.loc12_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_signed.loc13, %.loc13_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%int_32.loc14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_13.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_13.2: type = converted %int.make_type_signed.loc14, %.loc14_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   %.loc15_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: type = converted %.loc15_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %.loc16_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c.param_patt: %tuple.type.2) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc13: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc13: <specific function> = specific_function %Convert.bound.loc13, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.specific_fn.loc13(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_16.1: %i32 = value_of_initializer %int.convert_checked.loc13 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_16.2: %i32 = converted %int_1.loc13, %.loc13_16.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc13_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc14_25.1: %tuple.type.3 = tuple_literal (%int_1.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc14: <specific function> = specific_function %Convert.bound.loc14, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc14: init %i32 = call %Convert.specific_fn.loc14(%int_1.loc14) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_25.2: %i32 = value_of_initializer %int.convert_checked.loc14 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_25.3: %i32 = converted %int_1.loc14, %.loc14_25.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc14_25.3) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_25.4: %tuple.type.2 = converted %.loc14_25.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc14_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%int_1.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//api, inst+33, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//api, inst+58, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//api, inst+61, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+64, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//api, inst+65, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.12
// CHECK:STDOUT:     .ImplicitAs = %import_ref.13
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:     .B = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = imports.%import_ref.4
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc52_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc52_9.2: type = converted %.loc52_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc53: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc53: init type = call constants.%Int(%int_32.loc53) [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_8.1: type = value_of_initializer %int.make_type_signed.loc53 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_8.2: type = converted %int.make_type_signed.loc53, %.loc53_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc54: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc54: init type = call constants.%Int(%int_32.loc54) [template = constants.%i32]
// CHECK:STDOUT:   %.loc54_13.1: type = value_of_initializer %int.make_type_signed.loc54 [template = constants.%i32]
// CHECK:STDOUT:   %.loc54_13.2: type = converted %int.make_type_signed.loc54, %.loc54_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   %.loc55_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc55_9.2: type = converted %.loc55_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %.loc56_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc56_9.2: type = converted %.loc56_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c.param_patt: %tuple.type.1) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc53: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc53: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc53: <specific function> = specific_function %Convert.bound.loc53, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc53: init %i32 = call %Convert.specific_fn.loc53(%int_1.loc53) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc53_16.1: %i32 = value_of_initializer %int.convert_checked.loc53 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc53_16.2: %i32 = converted %int_1.loc53, %.loc53_16.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc53_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %int_1.loc54: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc54_25.1: %tuple.type.2 = tuple_literal (%int_1.loc54)
// CHECK:STDOUT:   %impl.elem0.loc54: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc54: <bound method> = bound_method %int_1.loc54, %impl.elem0.loc54 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc54: <specific function> = specific_function %Convert.bound.loc54, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc54: init %i32 = call %Convert.specific_fn.loc54(%int_1.loc54) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc54_25.2: %i32 = value_of_initializer %int.convert_checked.loc54 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc54_25.3: %i32 = converted %int_1.loc54, %.loc54_25.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.loc54_25.3) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc54_25.4: %tuple.type.1 = converted %.loc54_25.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc54_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_merge_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (%i32) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%int_1.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//extern_api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//extern_api, inst+33, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//extern_api, inst+58, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//extern_api, inst+61, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//extern_api, inst+64, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//extern_api, inst+65, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.12
// CHECK:STDOUT:     .ImplicitAs = %import_ref.13
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:     .B = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = imports.%import_ref.4
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc51_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc51_9.2: type = converted %.loc51_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc52: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc52: init type = call constants.%Int(%int_32.loc52) [template = constants.%i32]
// CHECK:STDOUT:   %.loc52_8.1: type = value_of_initializer %int.make_type_signed.loc52 [template = constants.%i32]
// CHECK:STDOUT:   %.loc52_8.2: type = converted %int.make_type_signed.loc52, %.loc52_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc53: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc53: init type = call constants.%Int(%int_32.loc53) [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_13.1: type = value_of_initializer %int.make_type_signed.loc53 [template = constants.%i32]
// CHECK:STDOUT:   %.loc53_13.2: type = converted %int.make_type_signed.loc53, %.loc53_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %i32} [template = constants.%struct_type.c]
// CHECK:STDOUT:   %c.var: ref %struct_type.c = var c
// CHECK:STDOUT:   %c: ref %struct_type.c = bind_name c, %c.var
// CHECK:STDOUT:   %.loc54_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc54_9.2: type = converted %.loc54_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %.loc55_9.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc55_9.2: type = converted %.loc55_9.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %e.var: ref %empty_tuple.type = var e
// CHECK:STDOUT:   %e: ref %empty_tuple.type = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b.param_patt: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c.param_patt: %tuple.type.1) -> %struct_type.c;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %int_1.loc52: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %impl.elem0.loc52: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc52: <bound method> = bound_method %int_1.loc52, %impl.elem0.loc52 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc52: <specific function> = specific_function %Convert.bound.loc52, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc52: init %i32 = call %Convert.specific_fn.loc52(%int_1.loc52) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc52_16.1: %i32 = value_of_initializer %int.convert_checked.loc52 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc52_16.2: %i32 = converted %int_1.loc52, %.loc52_16.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %B.call: init %i32 = call %B.ref(%.loc52_16.2)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %int_1.loc53: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc53_25.1: %tuple.type.2 = tuple_literal (%int_1.loc53)
// CHECK:STDOUT:   %impl.elem0.loc53: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc53: <bound method> = bound_method %int_1.loc53, %impl.elem0.loc53 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc53: <specific function> = specific_function %Convert.bound.loc53, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc53: init %i32 = call %Convert.specific_fn.loc53(%int_1.loc53) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc53_25.2: %i32 = value_of_initializer %int.convert_checked.loc53 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc53_25.3: %i32 = converted %int_1.loc53, %.loc53_25.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple: %tuple.type.1 = tuple_value (%.loc53_25.3) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc53_25.4: %tuple.type.1 = converted %.loc53_25.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %struct_type.c = call %C.ref(%.loc53_25.4)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %empty_tuple.type = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %empty_tuple.type = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//api, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//api, inst+33, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//api, inst+58, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//api, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+64, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:     .B = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = imports.%import_ref.4
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unloaded_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//extern_api, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//extern_api, inst+33, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//extern_api, inst+58, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//extern_api, inst+61, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//extern_api, inst+64, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = imports.%import_ref.1
// CHECK:STDOUT:     .B = imports.%import_ref.2
// CHECK:STDOUT:     .C = imports.%import_ref.3
// CHECK:STDOUT:     .D = imports.%import_ref.4
// CHECK:STDOUT:     .NS = imports.%NS
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
