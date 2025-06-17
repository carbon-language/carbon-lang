// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/fail_type_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/fail_type_param_mismatch.carbon

fn F(T:! type, U:! type) {
  var p: T*;
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+7]]:14: error: cannot implicitly convert expression of type `T` to `U` [ConversionFailure]
  // CHECK:STDERR:   let n: U = *p;
  // CHECK:STDERR:              ^~
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+4]]:14: note: type `T` does not implement interface `Core.ImplicitAs(U)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let n: U = *p;
  // CHECK:STDERR:              ^~
  // CHECK:STDERR:
  let n: U = *p;
}

// CHECK:STDOUT: --- fail_type_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %pattern_type.a32: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Dest: type = bind_symbolic_name Dest, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.275: type = fn_type @Convert, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %Convert.42e: %Convert.type.275 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.ca0: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%Dest) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.d88: type = facet_type <@ImplicitAs, @ImplicitAs(%U)> [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.5fe: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.a0b: %ImplicitAs.assoc_type.5fe = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic]
// CHECK:STDOUT:   %require_complete.6c8: <witness> = require_complete_type %ImplicitAs.type.d88 [symbolic]
// CHECK:STDOUT:   %assoc0.dc0: %ImplicitAs.assoc_type.ca0 = assoc_entity element0, imports.%Core.import_ref.207 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.492: @ImplicitAs.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.ca0) = import_ref Core//prelude/parts/as, loc10_35, loaded [symbolic = @ImplicitAs.%assoc0 (constants.%assoc0.dc0)]
// CHECK:STDOUT:   %Core.import_ref.1c7: @ImplicitAs.%Convert.type (%Convert.type.275) = import_ref Core//prelude/parts/as, loc10_35, loaded [symbolic = @ImplicitAs.%Convert (constants.%Convert.42e)]
// CHECK:STDOUT:   %Core.import_ref.207 = import_ref Core//prelude/parts/as, loc10_35, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc15_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc15_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_16.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc15_6.1: type, %U.loc15_16.1: type) {
// CHECK:STDOUT:   %T.loc15_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:   %U.loc15_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_16.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc16_11.2: type = ptr_type %T.loc15_6.2 [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %ptr.loc16_11.2 [symbolic = %require_complete.loc16 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %pattern_type.loc16: type = pattern_type %ptr.loc16_11.2 [symbolic = %pattern_type.loc16 (constants.%pattern_type.afe)]
// CHECK:STDOUT:   %require_complete.loc24_10: <witness> = require_complete_type %U.loc15_16.2 [symbolic = %require_complete.loc24_10 (constants.%require_complete.b54)]
// CHECK:STDOUT:   %pattern_type.loc24: type = pattern_type %U.loc15_16.2 [symbolic = %pattern_type.loc24 (constants.%pattern_type.a32)]
// CHECK:STDOUT:   %ImplicitAs.type.loc24_14.2: type = facet_type <@ImplicitAs, @ImplicitAs(%U.loc15_16.2)> [symbolic = %ImplicitAs.type.loc24_14.2 (constants.%ImplicitAs.type.d88)]
// CHECK:STDOUT:   %require_complete.loc24_14: <witness> = require_complete_type %ImplicitAs.type.loc24_14.2 [symbolic = %require_complete.loc24_14 (constants.%require_complete.6c8)]
// CHECK:STDOUT:   %ImplicitAs.assoc_type: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%U.loc15_16.2) [symbolic = %ImplicitAs.assoc_type (constants.%ImplicitAs.assoc_type.5fe)]
// CHECK:STDOUT:   %assoc0: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.5fe) = assoc_entity element0, imports.%Core.import_ref.1c7 [symbolic = %assoc0 (constants.%assoc0.a0b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %p.patt: @F.%pattern_type.loc16 (%pattern_type.afe) = binding_pattern p [concrete]
// CHECK:STDOUT:       %p.var_patt: @F.%pattern_type.loc16 (%pattern_type.afe) = var_pattern %p.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.var: ref @F.%ptr.loc16_11.2 (%ptr) = var %p.var_patt
// CHECK:STDOUT:     %.loc16: type = splice_block %ptr.loc16_11.1 [symbolic = %ptr.loc16_11.2 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc15_6.1 [symbolic = %T.loc15_6.2 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc16_11.1: type = ptr_type %T.ref [symbolic = %ptr.loc16_11.2 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: ref @F.%ptr.loc16_11.2 (%ptr) = bind_name p, %p.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: @F.%pattern_type.loc24 (%pattern_type.a32) = binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p.ref: ref @F.%ptr.loc16_11.2 (%ptr) = name_ref p, %p
// CHECK:STDOUT:     %.loc24_15: @F.%ptr.loc16_11.2 (%ptr) = bind_value %p.ref
// CHECK:STDOUT:     %.loc24_14.1: ref @F.%T.loc15_6.2 (%T) = deref %.loc24_15
// CHECK:STDOUT:     %U.ref: type = name_ref U, %U.loc15_16.1 [symbolic = %U.loc15_16.2 (constants.%U)]
// CHECK:STDOUT:     %ImplicitAs.type.loc24_14.1: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%U)> [symbolic = %ImplicitAs.type.loc24_14.2 (constants.%ImplicitAs.type.d88)]
// CHECK:STDOUT:     %.loc24_14.2: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.5fe) = specific_constant imports.%Core.import_ref.492, @ImplicitAs(constants.%U) [symbolic = %assoc0 (constants.%assoc0.a0b)]
// CHECK:STDOUT:     %Convert.ref: @F.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.5fe) = name_ref Convert, %.loc24_14.2 [symbolic = %assoc0 (constants.%assoc0.a0b)]
// CHECK:STDOUT:     %.loc24_14.3: @F.%U.loc15_16.2 (%U) = converted %.loc24_14.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %n: @F.%U.loc15_16.2 (%U) = bind_name n, <error> [concrete = <error>]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc15_16.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
