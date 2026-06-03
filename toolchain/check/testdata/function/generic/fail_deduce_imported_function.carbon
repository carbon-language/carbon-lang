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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/fail_deduce_imported_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/fail_deduce_imported_function.carbon

// --- lib.carbon
package Lib;

interface Z {}
fn A[T:! Z](unused x: {.a: T}) {}

// --- fail_deduce_imported_function.carbon

import Lib;

fn A[T:! Lib.Z](unused x: {.a: T}) {}

fn B() {
  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   A({.b = {}});
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! Lib.Z](unused x: {.a: T}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({.b = {}});

  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE+8]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   Lib.A({.b = {}});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: lib.carbon:4:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! Z](unused x: {.a: T}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Lib.A({.b = {}});
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.1d1: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type} [symbolic]
// CHECK:STDOUT:   %pattern_type.662: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %T.patt: %pattern_type.1d1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @A.%pattern_type (%pattern_type.662) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @A.%pattern_type (%pattern_type.662) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_10: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_7.2: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @A.%struct_type.a.loc4_29.1 (%struct_type.a) = value_param call_param0
// CHECK:STDOUT:     %.loc4_29: type = splice_block %struct_type.a.loc4_29.2 [symbolic = %struct_type.a.loc4_29.1 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref: %Z.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc4_28.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc4_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc4_28: type = converted %T.ref, %T.as_type.loc4_28.2 [symbolic = %T.as_type.loc4_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %struct_type.a.loc4_29.2: type = struct_type {.a: @A.%T.as_type.loc4_28.1 (%T.as_type)} [symbolic = %struct_type.a.loc4_29.1 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @A.%struct_type.a.loc4_29.1 (%struct_type.a) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z.WithSelf.decl = interface_with_self_decl @Z [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc4_7.2: %Z.type) {
// CHECK:STDOUT:   %T.loc4_7.1: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc4_28.1: type = facet_access_type %T.loc4_7.1 [symbolic = %T.as_type.loc4_28.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %struct_type.a.loc4_29.1: type = struct_type {.a: @A.%T.as_type.loc4_28.1 (%T.as_type)} [symbolic = %struct_type.a.loc4_29.1 (constants.%struct_type.a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.a.loc4_29.1 [symbolic = %pattern_type (constants.%pattern_type.662)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %struct_type.a.loc4_29.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @A.%struct_type.a.loc4_29.1 (%struct_type.a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc4_28.1 => constants.%T.as_type
// CHECK:STDOUT:   %struct_type.a.loc4_29.1 => constants.%struct_type.a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.662
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_imported_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.1d1: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type} [symbolic]
// CHECK:STDOUT:   %pattern_type.662: type = pattern_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %A.type.9f4: type = fn_type @A.loc4 [concrete]
// CHECK:STDOUT:   %A.921: %A.type.9f4 = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.b = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %A.type.42a: type = fn_type @A.1 [concrete]
// CHECK:STDOUT:   %A.b9a: %A.type.42a = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Lib: <namespace> = namespace file.%Lib.import, [concrete] {
// CHECK:STDOUT:     .Z = %Lib.Z
// CHECK:STDOUT:     .A = %Lib.A
// CHECK:STDOUT:     import Lib//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Lib.Z: type = import_ref Lib//default, Z, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Lib.import_ref.4cd = import_ref Lib//default, loc3_13, unloaded
// CHECK:STDOUT:   %Lib.A: %A.type.42a = import_ref Lib//default, A, loaded [concrete = constants.%A.b9a]
// CHECK:STDOUT:   %Lib.import_ref.78f: %Z.type = import_ref Lib//default, loc4_7, loaded [symbolic = @A.1.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Lib = imports.%Lib
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Lib.import = import Lib
// CHECK:STDOUT:   %A.decl: %A.type.9f4 = fn_decl @A.loc4 [concrete = constants.%A.921] {
// CHECK:STDOUT:     %T.patt: %pattern_type.1d1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @A.loc4.%pattern_type (%pattern_type.662) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @A.loc4.%pattern_type (%pattern_type.662) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Lib.ref: <namespace> = name_ref Lib, imports.%Lib [concrete = imports.%Lib]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, imports.%Lib.Z [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_7.2: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @A.loc4.%struct_type.a.loc4_33.1 (%struct_type.a) = value_param call_param0
// CHECK:STDOUT:     %.loc4_33: type = splice_block %struct_type.a.loc4_33.2 [symbolic = %struct_type.a.loc4_33.1 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref: %Z.type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc4_32.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc4_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc4_32: type = converted %T.ref, %T.as_type.loc4_32.2 [symbolic = %T.as_type.loc4_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %struct_type.a.loc4_33.2: type = struct_type {.a: @A.loc4.%T.as_type.loc4_32.1 (%T.as_type)} [symbolic = %struct_type.a.loc4_33.1 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @A.loc4.%struct_type.a.loc4_33.1 (%struct_type.a) = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Lib.import_ref.4cd
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.loc4(%T.loc4_7.2: %Z.type) {
// CHECK:STDOUT:   %T.loc4_7.1: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc4_32.1: type = facet_access_type %T.loc4_7.1 [symbolic = %T.as_type.loc4_32.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %struct_type.a.loc4_33.1: type = struct_type {.a: @A.loc4.%T.as_type.loc4_32.1 (%T.as_type)} [symbolic = %struct_type.a.loc4_33.1 (constants.%struct_type.a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.a.loc4_33.1 [symbolic = %pattern_type (constants.%pattern_type.662)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %struct_type.a.loc4_33.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @A.loc4.%struct_type.a.loc4_33.1 (%struct_type.a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc14: %A.type.9f4 = name_ref A, file.%A.decl [concrete = constants.%A.921]
// CHECK:STDOUT:   %.loc14_12: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc14_13: %struct_type.b = struct_literal (%.loc14_12) [concrete = constants.%struct]
// CHECK:STDOUT:   %Lib.ref: <namespace> = name_ref Lib, imports.%Lib [concrete = imports.%Lib]
// CHECK:STDOUT:   %A.ref.loc24: %A.type.42a = name_ref A, imports.%Lib.A [concrete = constants.%A.b9a]
// CHECK:STDOUT:   %.loc24_16: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_17: %struct_type.b = struct_literal (%.loc24_16) [concrete = constants.%struct]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.1(imports.%Lib.import_ref.78f: %Z.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @A.1.%T.as_type (%T.as_type)} [symbolic = %struct_type.a (constants.%struct_type.a)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.a [symbolic = %pattern_type (constants.%pattern_type.662)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %struct_type.a [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.loc4(constants.%T) {
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc4_32.1 => constants.%T.as_type
// CHECK:STDOUT:   %struct_type.a.loc4_33.1 => constants.%struct_type.a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.662
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %struct_type.a => constants.%struct_type.a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.662
// CHECK:STDOUT: }
// CHECK:STDOUT:
