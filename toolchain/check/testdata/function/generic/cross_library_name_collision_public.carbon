// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/cross_library_name_collision_public.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/cross_library_name_collision_public.carbon

// --- lib1.carbon

library "[[@TEST_NAME]]";

fn F[T:! type](a: T, b: T) -> T {
  return a;
}

fn Lib1CallF[T:! type](a: T, b: T) -> T {
  return F(a, b);
}

// --- lib2.carbon

library "[[@TEST_NAME]]";

// Duplicate function name in different files.
// This should be an error due to name conflict.
fn F[T:! type](a: T, b: T) -> T {
  return b;
}

fn Lib2CallF[T:! type](a: T, b: T) -> T {
  return F(a, b);
}

// --- fail_use.carbon

// TODO: If the calls to `Lib1CallF` and `Lib2CallF` are in different libraries,
// we will need a mechanism to detect this, presumably at link time.

// CHECK:STDERR: fail_use.carbon:[[@LINE+9]]:1: in import [InImport]
// CHECK:STDERR: lib2.carbon:6:1: error: duplicate name `F` being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: fn F[T:! type](a: T, b: T) -> T {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_use.carbon:[[@LINE+5]]:1: in import [InImport]
// CHECK:STDERR: lib1.carbon:4:1: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: fn F[T:! type](a: T, b: T) -> T {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "lib1";
import library "lib2";

fn Run() -> i32 {
  return Lib1CallF(1 as i32, 2 as i32) + Lib2CallF(1 as i32, 2 as i32);
}

// CHECK:STDOUT: --- lib1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Lib1CallF.type: type = fn_type @Lib1CallF [concrete]
// CHECK:STDOUT:   %Lib1CallF: %Lib1CallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Lib1CallF = %Lib1CallF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.7dc) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: @F.%pattern_type (%pattern_type.7dc) = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc4_31: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc4_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%T.loc4_6.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc4_19: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %a: @F.%T.loc4_6.2 (%T) = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: @F.%T.loc4_6.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc4_25: type = name_ref T, %T.loc4_6.1 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:     %b: @F.%T.loc4_6.2 (%T) = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc4_6.2 (%T) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.%T.loc4_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Lib1CallF.decl: %Lib1CallF.type = fn_decl @Lib1CallF [concrete = constants.%Lib1CallF] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @Lib1CallF.%pattern_type (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @Lib1CallF.%pattern_type (%pattern_type.7dc) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: @Lib1CallF.%pattern_type (%pattern_type.7dc) = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: @Lib1CallF.%pattern_type (%pattern_type.7dc) = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Lib1CallF.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Lib1CallF.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc8_39: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc8_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @Lib1CallF.%T.loc8_14.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc8_27: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %a: @Lib1CallF.%T.loc8_14.2 (%T) = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: @Lib1CallF.%T.loc8_14.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc8_33: type = name_ref T, %T.loc8_14.1 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:     %b: @Lib1CallF.%T.loc8_14.2 (%T) = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref @Lib1CallF.%T.loc8_14.2 (%T) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Lib1CallF.%T.loc8_14.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_6.1: type) {
// CHECK:STDOUT:   %T.loc4_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_6.2 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_6.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_6.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%T.loc4_6.2 (%T), %b.param: @F.%T.loc4_6.2 (%T)) -> @F.%T.loc4_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%T.loc4_6.2 (%T) = name_ref a, %a
// CHECK:STDOUT:     return %a.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Lib1CallF(%T.loc8_14.1: type) {
// CHECK:STDOUT:   %T.loc8_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_14.2 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_14.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc8_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc9_10.2: <specific function> = specific_function constants.%F, @F(%T.loc8_14.2) [symbolic = %F.specific_fn.loc9_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @Lib1CallF.%T.loc8_14.2 (%T), %b.param: @Lib1CallF.%T.loc8_14.2 (%T)) -> @Lib1CallF.%T.loc8_14.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @Lib1CallF.%T.loc8_14.2 (%T) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @Lib1CallF.%T.loc8_14.2 (%T) = name_ref b, %b
// CHECK:STDOUT:     %F.specific_fn.loc9_10.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc9_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init @Lib1CallF.%T.loc8_14.2 (%T) = call %F.specific_fn.loc9_10.1(%a.ref, %b.ref)
// CHECK:STDOUT:     %.loc9_17.1: @Lib1CallF.%T.loc8_14.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc9_17.2: @Lib1CallF.%T.loc8_14.2 (%T) = converted %F.call, %.loc9_17.1
// CHECK:STDOUT:     return %.loc9_17.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc4_6.2 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Lib1CallF(constants.%T) {
// CHECK:STDOUT:   %T.loc8_14.2 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- lib2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Lib2CallF.type: type = fn_type @Lib2CallF [concrete]
// CHECK:STDOUT:   %Lib2CallF: %Lib2CallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Lib2CallF = %Lib2CallF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.7dc) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: @F.%pattern_type (%pattern_type.7dc) = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: @F.%pattern_type (%pattern_type.7dc) = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_31: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%T.loc6_6.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc6_19: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a: @F.%T.loc6_6.2 (%T) = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: @F.%T.loc6_6.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc6_25: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %b: @F.%T.loc6_6.2 (%T) = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Lib2CallF.decl: %Lib2CallF.type = fn_decl @Lib2CallF [concrete = constants.%Lib2CallF] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @Lib2CallF.%pattern_type (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @Lib2CallF.%pattern_type (%pattern_type.7dc) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: @Lib2CallF.%pattern_type (%pattern_type.7dc) = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: @Lib2CallF.%pattern_type (%pattern_type.7dc) = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: @Lib2CallF.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Lib2CallF.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc10_39: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc10_14.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @Lib2CallF.%T.loc10_14.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc10_27: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %a: @Lib2CallF.%T.loc10_14.2 (%T) = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: @Lib2CallF.%T.loc10_14.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc10_33: type = name_ref T, %T.loc10_14.1 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:     %b: @Lib2CallF.%T.loc10_14.2 (%T) = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref @Lib2CallF.%T.loc10_14.2 (%T) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Lib2CallF.%T.loc10_14.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc6_6.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc6_6.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%T.loc6_6.2 (%T), %b.param: @F.%T.loc6_6.2 (%T)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %b.ref: @F.%T.loc6_6.2 (%T) = name_ref b, %b
// CHECK:STDOUT:     return %b.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Lib2CallF(%T.loc10_14.1: type) {
// CHECK:STDOUT:   %T.loc10_14.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc10_14.2 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc10_14.2 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc10_14.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc11_10.2: <specific function> = specific_function constants.%F, @F(%T.loc10_14.2) [symbolic = %F.specific_fn.loc11_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @Lib2CallF.%T.loc10_14.2 (%T), %b.param: @Lib2CallF.%T.loc10_14.2 (%T)) -> @Lib2CallF.%T.loc10_14.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @Lib2CallF.%T.loc10_14.2 (%T) = name_ref a, %a
// CHECK:STDOUT:     %b.ref: @Lib2CallF.%T.loc10_14.2 (%T) = name_ref b, %b
// CHECK:STDOUT:     %F.specific_fn.loc11_10.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc11_10.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %F.call: init @Lib2CallF.%T.loc10_14.2 (%T) = call %F.specific_fn.loc11_10.1(%a.ref, %b.ref)
// CHECK:STDOUT:     %.loc11_17.1: @Lib2CallF.%T.loc10_14.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc11_17.2: @Lib2CallF.%T.loc10_14.2 (%T) = converted %F.call, %.loc11_17.1
// CHECK:STDOUT:     return %.loc11_17.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Lib2CallF(constants.%T) {
// CHECK:STDOUT:   %T.loc10_14.2 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Lib1CallF.type: type = fn_type @Lib1CallF [concrete]
// CHECK:STDOUT:   %Lib1CallF: %Lib1CallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.type.b25846.1: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41931.1: %F.type.b25846.1 = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.ef173b.1: <specific function> = specific_function %F.c41931.1, @F.1(%T) [symbolic]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.1, @As(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.5, @impl.686(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %Add.type: type = facet_type <@Add> [concrete]
// CHECK:STDOUT:   %Op.type.52d: type = fn_type @Op.1, @impl.2c2(%N) [symbolic]
// CHECK:STDOUT:   %Op.12d: %Op.type.52d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.6b4: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%As.impl_witness.6b4) [concrete]
// CHECK:STDOUT:   %.982: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.c1b: <bound method> = bound_method %int_1.5b8, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.082: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.cdd: <bound method> = bound_method %int_2.ecc, %Convert.197 [concrete]
// CHECK:STDOUT:   %bound_method.78d: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Lib1CallF.specific_fn: <specific function> = specific_function %Lib1CallF, @Lib1CallF(%i32) [concrete]
// CHECK:STDOUT:   %Lib2CallF.type: type = fn_type @Lib2CallF [concrete]
// CHECK:STDOUT:   %Lib2CallF: %Lib2CallF.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.b25846.2: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41931.2: %F.type.b25846.2 = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.ef173b.2: <specific function> = specific_function %F.c41931.2, @F.2(%T) [symbolic]
// CHECK:STDOUT:   %Lib2CallF.specific_fn: <specific function> = specific_function %Lib2CallF, @Lib2CallF(%i32) [concrete]
// CHECK:STDOUT:   %Op.type.545: type = fn_type @Op.23 [concrete]
// CHECK:STDOUT:   %Add.impl_witness.dbc: <witness> = impl_witness imports.%Add.impl_witness_table, @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.type.210: type = fn_type @Op.1, @impl.2c2(%int_32) [concrete]
// CHECK:STDOUT:   %Op.c82: %Op.type.210 = struct_value () [concrete]
// CHECK:STDOUT:   %Add.facet: %Add.type = facet_value %i32, (%Add.impl_witness.dbc) [concrete]
// CHECK:STDOUT:   %.ac9: type = fn_type_with_self_type %Op.type.545, %Add.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.c82, @Op.1(%int_32) [concrete]
// CHECK:STDOUT:   %F.specific_fn.501db1.1: <specific function> = specific_function %F.c41931.1, @F.1(%i32) [concrete]
// CHECK:STDOUT:   %F.specific_fn.501db1.2: <specific function> = specific_function %F.c41931.2, @F.2(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F = import_ref Main//lib1, F, unloaded
// CHECK:STDOUT:   %Main.Lib1CallF: %Lib1CallF.type = import_ref Main//lib1, Lib1CallF, loaded [concrete = constants.%Lib1CallF]
// CHECK:STDOUT:   %Main.Lib2CallF: %Lib2CallF.type = import_ref Main//lib2, Lib2CallF, loaded [concrete = constants.%Lib2CallF]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Add = %Core.Add
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//lib1, loc8_14, loaded [symbolic = @Lib1CallF.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//lib1, loc4_6, loaded [symbolic = @F.1.%T (constants.%T)]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/operators/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.78a: @impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/types/int, loc28_39, loaded [symbolic = @impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @impl.686 [concrete]
// CHECK:STDOUT:   %Core.import_ref.344: @impl.2c2.%Op.type (%Op.type.52d) = import_ref Core//prelude/types/int, loc58_42, loaded [symbolic = @impl.2c2.%Op (constants.%Op.12d)]
// CHECK:STDOUT:   %Add.impl_witness_table = impl_witness_table (%Core.import_ref.344), @impl.2c2 [concrete]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//lib2, loc10_14, loaded [symbolic = @Lib2CallF.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.4: type = import_ref Main//lib2, loc6_6, loaded [symbolic = @F.2.%T (constants.%T)]
// CHECK:STDOUT:   %Core.Add: type = import_ref Core//prelude/operators/arithmetic, Add, loaded [concrete = constants.%Add.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Lib1CallF = imports.%Main.Lib1CallF
// CHECK:STDOUT:     .Lib2CallF = imports.%Main.Lib2CallF
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Lib1CallF.ref: %Lib1CallF.type = name_ref Lib1CallF, imports.%Main.Lib1CallF [concrete = constants.%Lib1CallF]
// CHECK:STDOUT:   %int_1.loc18_20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc18_25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc18_22: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc18_22.1: <bound method> = bound_method %int_1.loc18_20, %impl.elem0.loc18_22 [concrete = constants.%Convert.bound.c1b]
// CHECK:STDOUT:   %specific_fn.loc18_22: <specific function> = specific_function %impl.elem0.loc18_22, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_22.2: <bound method> = bound_method %int_1.loc18_20, %specific_fn.loc18_22 [concrete = constants.%bound_method.082]
// CHECK:STDOUT:   %int.convert_checked.loc18_22: init %i32 = call %bound_method.loc18_22.2(%int_1.loc18_20) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_22.1: %i32 = value_of_initializer %int.convert_checked.loc18_22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_22.2: %i32 = converted %int_1.loc18_20, %.loc18_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_2.loc18_30: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_32.loc18_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc18_32: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc18_32.1: <bound method> = bound_method %int_2.loc18_30, %impl.elem0.loc18_32 [concrete = constants.%Convert.bound.cdd]
// CHECK:STDOUT:   %specific_fn.loc18_32: <specific function> = specific_function %impl.elem0.loc18_32, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_32.2: <bound method> = bound_method %int_2.loc18_30, %specific_fn.loc18_32 [concrete = constants.%bound_method.78d]
// CHECK:STDOUT:   %int.convert_checked.loc18_32: init %i32 = call %bound_method.loc18_32.2(%int_2.loc18_30) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_32.1: %i32 = value_of_initializer %int.convert_checked.loc18_32 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_32.2: %i32 = converted %int_2.loc18_30, %.loc18_32.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %Lib1CallF.specific_fn: <specific function> = specific_function %Lib1CallF.ref, @Lib1CallF(constants.%i32) [concrete = constants.%Lib1CallF.specific_fn]
// CHECK:STDOUT:   %Lib1CallF.call: init %i32 = call %Lib1CallF.specific_fn(%.loc18_22.2, %.loc18_32.2)
// CHECK:STDOUT:   %Lib2CallF.ref: %Lib2CallF.type = name_ref Lib2CallF, imports.%Main.Lib2CallF [concrete = constants.%Lib2CallF]
// CHECK:STDOUT:   %int_1.loc18_52: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc18_57: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_57: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc18_54: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc18_54.1: <bound method> = bound_method %int_1.loc18_52, %impl.elem0.loc18_54 [concrete = constants.%Convert.bound.c1b]
// CHECK:STDOUT:   %specific_fn.loc18_54: <specific function> = specific_function %impl.elem0.loc18_54, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_54.2: <bound method> = bound_method %int_1.loc18_52, %specific_fn.loc18_54 [concrete = constants.%bound_method.082]
// CHECK:STDOUT:   %int.convert_checked.loc18_54: init %i32 = call %bound_method.loc18_54.2(%int_1.loc18_52) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_54.1: %i32 = value_of_initializer %int.convert_checked.loc18_54 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_54.2: %i32 = converted %int_1.loc18_52, %.loc18_54.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_2.loc18_62: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_32.loc18_67: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_67: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc18_64: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc18_64.1: <bound method> = bound_method %int_2.loc18_62, %impl.elem0.loc18_64 [concrete = constants.%Convert.bound.cdd]
// CHECK:STDOUT:   %specific_fn.loc18_64: <specific function> = specific_function %impl.elem0.loc18_64, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_64.2: <bound method> = bound_method %int_2.loc18_62, %specific_fn.loc18_64 [concrete = constants.%bound_method.78d]
// CHECK:STDOUT:   %int.convert_checked.loc18_64: init %i32 = call %bound_method.loc18_64.2(%int_2.loc18_62) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_64.1: %i32 = value_of_initializer %int.convert_checked.loc18_64 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_64.2: %i32 = converted %int_2.loc18_62, %.loc18_64.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %Lib2CallF.specific_fn: <specific function> = specific_function %Lib2CallF.ref, @Lib2CallF(constants.%i32) [concrete = constants.%Lib2CallF.specific_fn]
// CHECK:STDOUT:   %Lib2CallF.call: init %i32 = call %Lib2CallF.specific_fn(%.loc18_54.2, %.loc18_64.2)
// CHECK:STDOUT:   %impl.elem0.loc18_40: %.ac9 = impl_witness_access constants.%Add.impl_witness.dbc, element0 [concrete = constants.%Op.c82]
// CHECK:STDOUT:   %bound_method.loc18_40.1: <bound method> = bound_method %Lib1CallF.call, %impl.elem0.loc18_40
// CHECK:STDOUT:   %specific_fn.loc18_40: <specific function> = specific_function %impl.elem0.loc18_40, @Op.1(constants.%int_32) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_40.2: <bound method> = bound_method %Lib1CallF.call, %specific_fn.loc18_40
// CHECK:STDOUT:   %.loc18_38.1: %i32 = value_of_initializer %Lib1CallF.call
// CHECK:STDOUT:   %.loc18_38.2: %i32 = converted %Lib1CallF.call, %.loc18_38.1
// CHECK:STDOUT:   %.loc18_70.1: %i32 = value_of_initializer %Lib2CallF.call
// CHECK:STDOUT:   %.loc18_70.2: %i32 = converted %Lib2CallF.call, %.loc18_70.1
// CHECK:STDOUT:   %int.sadd: init %i32 = call %bound_method.loc18_40.2(%.loc18_38.2, %.loc18_70.2)
// CHECK:STDOUT:   %.loc18_71.1: %i32 = value_of_initializer %int.sadd
// CHECK:STDOUT:   %.loc18_71.2: %i32 = converted %int.sadd, %.loc18_71.1
// CHECK:STDOUT:   return %.loc18_71.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Lib1CallF(imports.%Main.import_ref.5ab3ec.1: type) [from "lib1.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function constants.%F.c41931.1, @F.1(%T) [symbolic = %F.specific_fn (constants.%F.specific_fn.ef173b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(imports.%Main.import_ref.5ab3ec.2: type) [from "lib1.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Lib2CallF(imports.%Main.import_ref.5ab3ec.3: type) [from "lib2.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function constants.%F.c41931.2, @F.2(%T) [symbolic = %F.specific_fn (constants.%F.specific_fn.ef173b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(imports.%Main.import_ref.5ab3ec.4: type) [from "lib2.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Lib1CallF(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Lib1CallF(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %F.specific_fn => constants.%F.specific_fn.501db1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Lib2CallF(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Lib2CallF(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %F.specific_fn => constants.%F.specific_fn.501db1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
