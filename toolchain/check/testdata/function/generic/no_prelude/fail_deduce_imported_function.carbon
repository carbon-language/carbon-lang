// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/fail_deduce_imported_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/fail_deduce_imported_function.carbon

// --- lib.carbon
package Lib;

interface Z {}
fn A[T:! Z](x: {.a: T}) {}

// --- fail_deduce_imported_function.carbon

import Lib;

fn A[T:! Lib.Z](x: {.a: T}) {}

fn B() {
  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   A({.b = {}});
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! Lib.Z](x: {.a: T}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  A({.b = {}});

  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE+8]]:3: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   Lib.A({.b = {}});
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_deduce_imported_function.carbon:[[@LINE-17]]:1: in import [InImport]
  // CHECK:STDERR: lib.carbon:4:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn A[T:! Z](x: {.a: T}) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Lib.A({.b = {}});
}

// CHECK:STDOUT: --- lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Z.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type} [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_6.1: %Z.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @A.%struct_type.a.loc4_22.2 (%struct_type.a) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @A.%struct_type.a.loc4_22.2 (%struct_type.a) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %T.loc4_6.1: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @A.%struct_type.a.loc4_22.2 (%struct_type.a) = value_param call_param0
// CHECK:STDOUT:     %.loc4_22: type = splice_block %struct_type.a.loc4_22.1 [symbolic = %struct_type.a.loc4_22.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref: %Z.type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc4_21.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc4_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc4_21: type = converted %T.ref, %T.as_type.loc4_21.1 [symbolic = %T.as_type.loc4_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %struct_type.a.loc4_22.1: type = struct_type {.a: %T.as_type} [symbolic = %struct_type.a.loc4_22.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @A.%struct_type.a.loc4_22.2 (%struct_type.a) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A(%T.loc4_6.1: %Z.type) {
// CHECK:STDOUT:   %T.loc4_6.2: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_6.2: %Z.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc4_21.2: type = facet_access_type %T.loc4_6.2 [symbolic = %T.as_type.loc4_21.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %struct_type.a.loc4_22.2: type = struct_type {.a: @A.%T.as_type.loc4_21.2 (%T.as_type)} [symbolic = %struct_type.a.loc4_22.2 (constants.%struct_type.a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.%struct_type.a.loc4_22.2 (%struct_type.a) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_6.1: %Z.type](%x.param_patt: @A.%struct_type.a.loc4_22.2 (%struct_type.a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_6.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc4_21.2 => constants.%T.as_type
// CHECK:STDOUT:   %struct_type.a.loc4_22.2 => constants.%struct_type.a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_deduce_imported_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %T: %Z.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: %Z.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T.as_type} [symbolic]
// CHECK:STDOUT:   %A.type.00d: type = fn_type @A.1 [concrete]
// CHECK:STDOUT:   %A.1db: %A.type.00d = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %struct_type.a [symbolic]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %A.type.fad: type = fn_type @A.2 [concrete]
// CHECK:STDOUT:   %A.7a0: %A.type.fad = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Lib: <namespace> = namespace file.%Lib.import, [concrete] {
// CHECK:STDOUT:     .Z = %Lib.Z
// CHECK:STDOUT:     .A = %Lib.A
// CHECK:STDOUT:     import Lib//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Lib.Z: type = import_ref Lib//default, Z, loaded [concrete = constants.%Z.type]
// CHECK:STDOUT:   %Lib.import_ref.f88 = import_ref Lib//default, inst17 [no loc], unloaded
// CHECK:STDOUT:   %Lib.A: %A.type.fad = import_ref Lib//default, A, loaded [concrete = constants.%A.7a0]
// CHECK:STDOUT:   %Lib.import_ref.be7: %Z.type = import_ref Lib//default, loc4_6, loaded [symbolic = @A.2.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Lib = imports.%Lib
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Lib.import = import Lib
// CHECK:STDOUT:   %A.decl: %A.type.00d = fn_decl @A.1 [concrete = constants.%A.1db] {
// CHECK:STDOUT:     %T.patt.loc4_6.1: %Z.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @A.1.%struct_type.a.loc4_26.2 (%struct_type.a) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @A.1.%struct_type.a.loc4_26.2 (%struct_type.a) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       %Lib.ref: <namespace> = name_ref Lib, imports.%Lib [concrete = imports.%Lib]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, imports.%Lib.Z [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_6.1: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @A.1.%struct_type.a.loc4_26.2 (%struct_type.a) = value_param call_param0
// CHECK:STDOUT:     %.loc4_26: type = splice_block %struct_type.a.loc4_26.1 [symbolic = %struct_type.a.loc4_26.2 (constants.%struct_type.a)] {
// CHECK:STDOUT:       %T.ref: %Z.type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc4_25.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc4_25.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc4_25: type = converted %T.ref, %T.as_type.loc4_25.1 [symbolic = %T.as_type.loc4_25.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %struct_type.a.loc4_26.1: type = struct_type {.a: %T.as_type} [symbolic = %struct_type.a.loc4_26.2 (constants.%struct_type.a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @A.1.%struct_type.a.loc4_26.2 (%struct_type.a) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z [from "lib.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Lib.import_ref.f88
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.1(%T.loc4_6.1: %Z.type) {
// CHECK:STDOUT:   %T.loc4_6.2: %Z.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_6.2: %Z.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type.loc4_25.2: type = facet_access_type %T.loc4_6.2 [symbolic = %T.as_type.loc4_25.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %struct_type.a.loc4_26.2: type = struct_type {.a: @A.1.%T.as_type.loc4_25.2 (%T.as_type)} [symbolic = %struct_type.a.loc4_26.2 (constants.%struct_type.a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.1.%struct_type.a.loc4_26.2 (%struct_type.a) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc4_6.1: %Z.type](%x.param_patt: @A.1.%struct_type.a.loc4_26.2 (%struct_type.a)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref.loc14: %A.type.00d = name_ref A, file.%A.decl [concrete = constants.%A.1db]
// CHECK:STDOUT:   %.loc14_12: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_13: %struct_type.b = struct_literal (%.loc14_12)
// CHECK:STDOUT:   %Lib.ref: <namespace> = name_ref Lib, imports.%Lib [concrete = imports.%Lib]
// CHECK:STDOUT:   %A.ref.loc24: %A.type.fad = name_ref A, imports.%Lib.A [concrete = constants.%A.7a0]
// CHECK:STDOUT:   %.loc24_16: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc24_17: %struct_type.b = struct_literal (%.loc24_16)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.2(imports.%Lib.import_ref.be7: %Z.type) [from "lib.carbon"] {
// CHECK:STDOUT:   %T: %Z.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt.2: %Z.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @A.2.%T.as_type (%T.as_type)} [symbolic = %struct_type.a (constants.%struct_type.a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @A.2.%struct_type.a (%struct_type.a) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.1: %Z.type](%x.param_patt: @A.2.%struct_type.a (%struct_type.a));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_6.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type.loc4_25.2 => constants.%T.as_type
// CHECK:STDOUT:   %struct_type.a.loc4_26.2 => constants.%struct_type.a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt.2 => constants.%T.patt
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %struct_type.a => constants.%struct_type.a
// CHECK:STDOUT: }
// CHECK:STDOUT:
