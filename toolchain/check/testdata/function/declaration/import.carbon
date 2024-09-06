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

library "api";

fn A();
fn B(b: i32) -> i32;
fn C(c: (i32,)) -> {.c: i32};
extern fn D();

namespace NS;
fn NS.E();

// --- extern_api.carbon

library "extern_api";

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

library "basics";

import library "api";

var a: () = A();
var b: i32 = B(1);
var c: {.c: i32} = C((1,));
var d: () = D();
var e: () = NS.E();

// --- fail_redecl_api.carbon

library "redecl_api";

import library "api";

// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+10]]:1: ERROR: Redeclarations of `fn A` must match use of `extern`.
// CHECK:STDERR: extern fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-5]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:4:1: Previously declared here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
extern fn A();
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+10]]:1: ERROR: Redeclarations of `fn B` must match use of `extern`.
// CHECK:STDERR: extern fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-16]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:5:1: Previously declared here.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn B(b: i32) -> i32;
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+10]]:1: ERROR: Redeclarations of `fn C` must match use of `extern`.
// CHECK:STDERR: extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-27]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:6:1: Previously declared here.
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+10]]:1: ERROR: Redeclaration of `fn D` is redundant.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-38]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:7:1: Previously declared here.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn D();
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE+10]]:1: ERROR: Redeclarations of `fn E` must match use of `extern`.
// CHECK:STDERR: extern fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl_api.carbon:[[@LINE-49]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:10:1: Previously declared here.
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

library "redecl_extern_api";

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

library "merge";

// CHECK:STDERR: fail_merge.carbon:[[@LINE+65]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+59]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+52]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:5:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+46]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:5:1: Name is previously declared here.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+39]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:6:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+33]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:6:1: Name is previously declared here.
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+26]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:7:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+20]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:7:1: Name is previously declared here.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge.carbon:[[@LINE+13]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:10:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern library "redecl_extern_api" fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge.carbon:[[@LINE+7]]:1: In import.
// CHECK:STDERR: import library "api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:10:1: Name is previously declared here.
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

library "merge_reverse";

// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+64]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:4:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+58]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:4:1: Name is previously declared here.
// CHECK:STDERR: extern library "redecl_extern_api" fn A();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+51]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:5:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+45]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:5:1: Name is previously declared here.
// CHECK:STDERR: extern library "redecl_extern_api" fn B(b: i32) -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+38]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:6:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+32]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:6:1: Name is previously declared here.
// CHECK:STDERR: extern library "redecl_extern_api" fn C(c: (i32,)) -> {.c: i32};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+25]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:7:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: extern fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+19]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:7:1: Name is previously declared here.
// CHECK:STDERR: extern library "redecl_extern_api" fn D();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+12]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: api.carbon:10:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn NS.E();
// CHECK:STDERR: ^~~~~~~~~~
// CHECK:STDERR: fail_merge_reverse.carbon:[[@LINE+6]]:1: In import.
// CHECK:STDERR: import library "extern_api";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: extern_api.carbon:10:1: Name is previously declared here.
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

library "unloaded";

import library "api";

// --- unloaded_extern.carbon

library "unloaded_extern";

import library "extern_api";

// CHECK:STDOUT: --- api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
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
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc5_9: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_9.1: type = value_of_initializer %int.make_type_32.loc5_9 [template = i32]
// CHECK:STDOUT:     %.loc5_9.2: type = converted %int.make_type_32.loc5_9, %.loc5_9.1 [template = i32]
// CHECK:STDOUT:     %b.loc5_6.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_6.1
// CHECK:STDOUT:     %int.make_type_32.loc5_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_17.1: type = value_of_initializer %int.make_type_32.loc5_17 [template = i32]
// CHECK:STDOUT:     %.loc5_17.2: type = converted %int.make_type_32.loc5_17, %.loc5_17.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc6_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_14.1: %.2 = tuple_literal (%int.make_type_32.loc6_10)
// CHECK:STDOUT:     %.loc6_14.2: type = value_of_initializer %int.make_type_32.loc6_10 [template = i32]
// CHECK:STDOUT:     %.loc6_14.3: type = converted %int.make_type_32.loc6_10, %.loc6_14.2 [template = i32]
// CHECK:STDOUT:     %.loc6_14.4: type = converted %.loc6_14.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc6_6.1: %.3 = param c
// CHECK:STDOUT:     @C.%c: %.3 = bind_name c, %c.loc6_6.1
// CHECK:STDOUT:     %int.make_type_32.loc6_25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_25.1: type = value_of_initializer %int.make_type_32.loc6_25 [template = i32]
// CHECK:STDOUT:     %.loc6_25.2: type = converted %int.make_type_32.loc6_25, %.loc6_25.1 [template = i32]
// CHECK:STDOUT:     %.loc6_28: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     @C.%return: ref %.4 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: %.3) -> %.4;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
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
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc5_44: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_44.1: type = value_of_initializer %int.make_type_32.loc5_44 [template = i32]
// CHECK:STDOUT:     %.loc5_44.2: type = converted %int.make_type_32.loc5_44, %.loc5_44.1 [template = i32]
// CHECK:STDOUT:     %b.loc5_41.1: i32 = param b
// CHECK:STDOUT:     @B.%b: i32 = bind_name b, %b.loc5_41.1
// CHECK:STDOUT:     %int.make_type_32.loc5_52: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc5_52.1: type = value_of_initializer %int.make_type_32.loc5_52 [template = i32]
// CHECK:STDOUT:     %.loc5_52.2: type = converted %int.make_type_32.loc5_52, %.loc5_52.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc6_45: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_49.1: %.2 = tuple_literal (%int.make_type_32.loc6_45)
// CHECK:STDOUT:     %.loc6_49.2: type = value_of_initializer %int.make_type_32.loc6_45 [template = i32]
// CHECK:STDOUT:     %.loc6_49.3: type = converted %int.make_type_32.loc6_45, %.loc6_49.2 [template = i32]
// CHECK:STDOUT:     %.loc6_49.4: type = converted %.loc6_49.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc6_41.1: %.3 = param c
// CHECK:STDOUT:     @C.%c: %.3 = bind_name c, %c.loc6_41.1
// CHECK:STDOUT:     %int.make_type_32.loc6_60: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_60.1: type = value_of_initializer %int.make_type_32.loc6_60 [template = i32]
// CHECK:STDOUT:     %.loc6_60.2: type = converted %int.make_type_32.loc6_60, %.loc6_60.1 [template = i32]
// CHECK:STDOUT:     %.loc6_63: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     @C.%return: ref %.4 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c: %.3) -> %.4;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//api, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//api, inst+41, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//api, inst+44, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+47, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//api, inst+48, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %.loc6_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc6_9.2: type = converted %.loc6_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc7: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc7_8.1: type = value_of_initializer %int.make_type_32.loc7 [template = i32]
// CHECK:STDOUT:   %.loc7_8.2: type = converted %int.make_type_32.loc7, %.loc7_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc8: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc8_13.1: type = value_of_initializer %int.make_type_32.loc8 [template = i32]
// CHECK:STDOUT:   %.loc8_13.2: type = converted %int.make_type_32.loc8, %.loc8_13.1 [template = i32]
// CHECK:STDOUT:   %.loc8_16: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref %.3 = var c
// CHECK:STDOUT:   %c: ref %.3 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc9_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc9_9.2: type = converted %.loc9_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref %.1 = var d
// CHECK:STDOUT:   %d: ref %.1 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc10_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc10_9.2: type = converted %.loc10_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref %.1 = var e
// CHECK:STDOUT:   %e: ref %.1 = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: %.4) -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.1 = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc7)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %.loc8_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25: %.4 = tuple_literal (%.loc8_23)
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.loc8_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc8_21: %.4 = converted %.loc8_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %.3 = call %C.ref(%.loc8_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %.1 = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %.1 = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//api, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//api, inst+41, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//api, inst+44, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+47, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = file.%E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//api, inst+48, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     .b = %b.loc63
// CHECK:STDOUT:     .c = %c.loc64
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc27_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc27_16.1: type = value_of_initializer %int.make_type_32.loc27_16 [template = i32]
// CHECK:STDOUT:     %.loc27_16.2: type = converted %int.make_type_32.loc27_16, %.loc27_16.1 [template = i32]
// CHECK:STDOUT:     %b.loc27_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc27_13.2: i32 = bind_name b, %b.loc27_13.1
// CHECK:STDOUT:     %int.make_type_32.loc27_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc27_24.1: type = value_of_initializer %int.make_type_32.loc27_24 [template = i32]
// CHECK:STDOUT:     %.loc27_24.2: type = converted %int.make_type_32.loc27_24, %.loc27_24.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc27: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc38_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc38_21.1: %.2 = tuple_literal (%int.make_type_32.loc38_17)
// CHECK:STDOUT:     %.loc38_21.2: type = value_of_initializer %int.make_type_32.loc38_17 [template = i32]
// CHECK:STDOUT:     %.loc38_21.3: type = converted %int.make_type_32.loc38_17, %.loc38_21.2 [template = i32]
// CHECK:STDOUT:     %.loc38_21.4: type = converted %.loc38_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc38_13.1: %.3 = param c
// CHECK:STDOUT:     %c.loc38_13.2: %.3 = bind_name c, %c.loc38_13.1
// CHECK:STDOUT:     %int.make_type_32.loc38_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc38_32.1: type = value_of_initializer %int.make_type_32.loc38_32 [template = i32]
// CHECK:STDOUT:     %.loc38_32.2: type = converted %int.make_type_32.loc38_32, %.loc38_32.1 [template = i32]
// CHECK:STDOUT:     %.loc38_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     %return.var.loc38: ref %.4 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT:   %.loc62_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc62_9.2: type = converted %.loc62_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc63: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc63_8.1: type = value_of_initializer %int.make_type_32.loc63 [template = i32]
// CHECK:STDOUT:   %.loc63_8.2: type = converted %int.make_type_32.loc63, %.loc63_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc63: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc64: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc64_13.1: type = value_of_initializer %int.make_type_32.loc64 [template = i32]
// CHECK:STDOUT:   %.loc64_13.2: type = converted %int.make_type_32.loc64, %.loc64_13.1 [template = i32]
// CHECK:STDOUT:   %.loc64_16: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:   %c.var: ref %.4 = var c
// CHECK:STDOUT:   %c.loc64: ref %.4 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc65_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc65_9.2: type = converted %.loc65_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref %.1 = var d
// CHECK:STDOUT:   %d: ref %.1 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc66_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc66_9.2: type = converted %.loc66_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref %.1 = var e
// CHECK:STDOUT:   %e: ref %.1 = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: %.3) -> %.4;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.1 = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc63: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc63)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc64_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc64_25: %.3 = tuple_literal (%.loc64_23)
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.loc64_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc64_21: %.3 = converted %.loc64_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %.4 = call %C.ref(%.loc64_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %.1 = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %.1 = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_extern_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.4: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//extern_api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//extern_api, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//extern_api, inst+41, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//extern_api, inst+44, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//extern_api, inst+47, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = file.%E.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//extern_api, inst+48, loaded [template = constants.%E]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.7
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.7: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     .b = %b.loc13
// CHECK:STDOUT:     .c = %c.loc14
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .e = %e
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc7_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_16.1: type = value_of_initializer %int.make_type_32.loc7_16 [template = i32]
// CHECK:STDOUT:     %.loc7_16.2: type = converted %int.make_type_32.loc7_16, %.loc7_16.1 [template = i32]
// CHECK:STDOUT:     %b.loc7_13.1: i32 = param b
// CHECK:STDOUT:     %b.loc7_13.2: i32 = bind_name b, %b.loc7_13.1
// CHECK:STDOUT:     %int.make_type_32.loc7_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc7_24.1: type = value_of_initializer %int.make_type_32.loc7_24 [template = i32]
// CHECK:STDOUT:     %.loc7_24.2: type = converted %int.make_type_32.loc7_24, %.loc7_24.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = fn_decl @C [template = constants.%C] {
// CHECK:STDOUT:     %int.make_type_32.loc8_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_21.1: %.2 = tuple_literal (%int.make_type_32.loc8_17)
// CHECK:STDOUT:     %.loc8_21.2: type = value_of_initializer %int.make_type_32.loc8_17 [template = i32]
// CHECK:STDOUT:     %.loc8_21.3: type = converted %int.make_type_32.loc8_17, %.loc8_21.2 [template = i32]
// CHECK:STDOUT:     %.loc8_21.4: type = converted %.loc8_21.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     %c.loc8_13.1: %.3 = param c
// CHECK:STDOUT:     %c.loc8_13.2: %.3 = bind_name c, %c.loc8_13.1
// CHECK:STDOUT:     %int.make_type_32.loc8_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_32.1: type = value_of_initializer %int.make_type_32.loc8_32 [template = i32]
// CHECK:STDOUT:     %.loc8_32.2: type = converted %int.make_type_32.loc8_32, %.loc8_32.1 [template = i32]
// CHECK:STDOUT:     %.loc8_35: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:     %return.var.loc8: ref %.4 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: %D.type = fn_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %E.decl: %E.type = fn_decl @E [template = constants.%E] {}
// CHECK:STDOUT:   %.loc12_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_32.loc13, %.loc13_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b.loc13: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_13.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_13.2: type = converted %int.make_type_32.loc14, %.loc14_13.1 [template = i32]
// CHECK:STDOUT:   %.loc14_16: type = struct_type {.c: i32} [template = constants.%.4]
// CHECK:STDOUT:   %c.var: ref %.4 = var c
// CHECK:STDOUT:   %c.loc14: ref %.4 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc15_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc15_9.2: type = converted %.loc15_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref %.1 = var d
// CHECK:STDOUT:   %d: ref %.1 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc16_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc16_9.2: type = converted %.loc16_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref %.1 = var e
// CHECK:STDOUT:   %e: ref %.1 = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c: %.3) -> %.4;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.1 = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc13)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc14_23: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: %.3 = tuple_literal (%.loc14_23)
// CHECK:STDOUT:   %tuple: %.3 = tuple_value (%.loc14_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_21: %.3 = converted %.loc14_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %.4 = call %C.ref(%.loc14_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %.1 = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, file.%E.decl [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %.1 = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_merge.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//api, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//api, inst+41, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//api, inst+44, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+47, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//api, inst+48, loaded [template = constants.%E]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//extern_api, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//extern_api, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//extern_api, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//extern_api, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//extern_api, inst+48, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.12
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.12: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %.loc72_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc72_9.2: type = converted %.loc72_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc73: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc73_8.1: type = value_of_initializer %int.make_type_32.loc73 [template = i32]
// CHECK:STDOUT:   %.loc73_8.2: type = converted %int.make_type_32.loc73, %.loc73_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc74: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc74_13.1: type = value_of_initializer %int.make_type_32.loc74 [template = i32]
// CHECK:STDOUT:   %.loc74_13.2: type = converted %int.make_type_32.loc74, %.loc74_13.1 [template = i32]
// CHECK:STDOUT:   %.loc74_16: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref %.3 = var c
// CHECK:STDOUT:   %c: ref %.3 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc75_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc75_9.2: type = converted %.loc75_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref %.1 = var d
// CHECK:STDOUT:   %d: ref %.1 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc76_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc76_9.2: type = converted %.loc76_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref %.1 = var e
// CHECK:STDOUT:   %e: ref %.1 = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%c: %.4) -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.1 = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %.loc73: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc73)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %.loc74_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc74_25: %.4 = tuple_literal (%.loc74_23)
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.loc74_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc74_21: %.4 = converted %.loc74_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %.3 = call %C.ref(%.loc74_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %.1 = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %.1 = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_merge_reverse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.c: i32} [template]
// CHECK:STDOUT:   %C.type: type = fn_type @C [template]
// CHECK:STDOUT:   %C: %C.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.2) [template]
// CHECK:STDOUT:   %D.type: type = fn_type @D [template]
// CHECK:STDOUT:   %D: %D.type = struct_value () [template]
// CHECK:STDOUT:   %E.type: type = fn_type @E [template]
// CHECK:STDOUT:   %E: %E.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: %A.type = import_ref Main//extern_api, inst+3, loaded [template = constants.%A]
// CHECK:STDOUT:   %import_ref.2: %B.type = import_ref Main//extern_api, inst+21, loaded [template = constants.%B]
// CHECK:STDOUT:   %import_ref.3: %C.type = import_ref Main//extern_api, inst+41, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.4: %D.type = import_ref Main//extern_api, inst+44, loaded [template = constants.%D]
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//extern_api, inst+47, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6: %E.type = import_ref Main//extern_api, inst+48, loaded [template = constants.%E]
// CHECK:STDOUT:   %import_ref.7 = import_ref Main//api, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.8 = import_ref Main//api, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.9 = import_ref Main//api, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.10 = import_ref Main//api, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.11 = import_ref Main//api, inst+48, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.12
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.12: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %.loc71_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc71_9.2: type = converted %.loc71_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc72: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc72_8.1: type = value_of_initializer %int.make_type_32.loc72 [template = i32]
// CHECK:STDOUT:   %.loc72_8.2: type = converted %int.make_type_32.loc72, %.loc72_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.make_type_32.loc73: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc73_13.1: type = value_of_initializer %int.make_type_32.loc73 [template = i32]
// CHECK:STDOUT:   %.loc73_13.2: type = converted %int.make_type_32.loc73, %.loc73_13.1 [template = i32]
// CHECK:STDOUT:   %.loc73_16: type = struct_type {.c: i32} [template = constants.%.3]
// CHECK:STDOUT:   %c.var: ref %.3 = var c
// CHECK:STDOUT:   %c: ref %.3 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc74_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc74_9.2: type = converted %.loc74_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref %.1 = var d
// CHECK:STDOUT:   %d: ref %.1 = bind_name d, %d.var
// CHECK:STDOUT:   %.loc75_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc75_9.2: type = converted %.loc75_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %e.var: ref %.1 = var e
// CHECK:STDOUT:   %e: ref %.1 = bind_name e, %e.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @B(%b: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @C(%c: %.4) -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @D();
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @E();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, imports.%import_ref.1 [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.1 = call %A.ref()
// CHECK:STDOUT:   assign file.%a.var, %A.call
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, imports.%import_ref.2 [template = constants.%B]
// CHECK:STDOUT:   %.loc72: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref(%.loc72)
// CHECK:STDOUT:   assign file.%b.var, %B.call
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.3 [template = constants.%C]
// CHECK:STDOUT:   %.loc73_23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc73_25: %.4 = tuple_literal (%.loc73_23)
// CHECK:STDOUT:   %tuple: %.4 = tuple_value (%.loc73_23) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc73_21: %.4 = converted %.loc73_25, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %C.call: init %.3 = call %C.ref(%.loc73_21)
// CHECK:STDOUT:   assign file.%c.var, %C.call
// CHECK:STDOUT:   %D.ref: %D.type = name_ref D, imports.%import_ref.4 [template = constants.%D]
// CHECK:STDOUT:   %D.call: init %.1 = call %D.ref()
// CHECK:STDOUT:   assign file.%d.var, %D.call
// CHECK:STDOUT:   %NS.ref: <namespace> = name_ref NS, imports.%NS [template = imports.%NS]
// CHECK:STDOUT:   %E.ref: %E.type = name_ref E, imports.%import_ref.6 [template = constants.%E]
// CHECK:STDOUT:   %E.call: init %.1 = call %E.ref()
// CHECK:STDOUT:   assign file.%e.var, %E.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- unloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1 = import_ref Main//api, inst+3, unloaded
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//api, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//api, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//api, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//api, inst+47, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//api, inst+48, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//extern_api, inst+21, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Main//extern_api, inst+41, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//extern_api, inst+44, unloaded
// CHECK:STDOUT:   %import_ref.5: <namespace> = import_ref Main//extern_api, inst+47, loaded
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref.5, [template] {
// CHECK:STDOUT:     .E = %import_ref.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.6 = import_ref Main//extern_api, inst+48, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
