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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/nested_struct_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/nested_struct_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var unused v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}

// CHECK:STDOUT: --- nested_struct_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.05f: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %.fd3: Core.Form = init_form %tuple.type.05f [concrete]
// CHECK:STDOUT:   %pattern_type.776: type = pattern_type %tuple.type.05f [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.120: type = struct_type {.a: %tuple.type.05f, .b: %tuple.type.05f} [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.f0c: type = pattern_type %struct_type.a.b.120 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc18_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.776 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.776 = return_slot_pattern %return.param_patt, %.loc15_25.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc15_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_25.1: %tuple.type.ff9 = tuple_literal (%i32.loc15_12, %i32.loc15_17, %i32.loc15_22) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc15_25.2: type = converted %.loc15_25.1, constants.%tuple.type.05f [concrete = constants.%tuple.type.05f]
// CHECK:STDOUT:     %.loc15_25.3: Core.Form = init_form %.loc15_25.2 [concrete = constants.%.fd3]
// CHECK:STDOUT:     %return.param: ref %tuple.type.05f = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.05f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %tuple.type.05f;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.f0c = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.f0c = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %struct_type.a.b.120 = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc18_68: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_81.1: ref %tuple.type.05f = struct_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc18_70: init %tuple.type.05f to %.loc18_81.1 = call %F.ref.loc18_68()
// CHECK:STDOUT:   %F.ref.loc18_78: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_81.2: ref %tuple.type.05f = struct_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc18_80: init %tuple.type.05f to %.loc18_81.2 = call %F.ref.loc18_78()
// CHECK:STDOUT:   %.loc18_81.3: %struct_type.a.b.120 = struct_literal (%F.call.loc18_70, %F.call.loc18_80)
// CHECK:STDOUT:   %.loc18_81.4: init %struct_type.a.b.120 to %v.var = struct_init (%F.call.loc18_70, %F.call.loc18_80)
// CHECK:STDOUT:   %.loc18_3: init %struct_type.a.b.120 = converted %.loc18_81.3, %.loc18_81.4
// CHECK:STDOUT:   assign %v.var, %.loc18_3
// CHECK:STDOUT:   %.loc18_58: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.120] {
// CHECK:STDOUT:     %i32.loc18_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.1: %tuple.type.ff9 = tuple_literal (%i32.loc18_23, %i32.loc18_28, %i32.loc18_33) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc18_36.2: type = converted %.loc18_36.1, constants.%tuple.type.05f [concrete = constants.%tuple.type.05f]
// CHECK:STDOUT:     %i32.loc18_44: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_54: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_57.1: %tuple.type.ff9 = tuple_literal (%i32.loc18_44, %i32.loc18_49, %i32.loc18_54) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc18_57.2: type = converted %.loc18_57.1, constants.%tuple.type.05f [concrete = constants.%tuple.type.05f]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %tuple.type.05f, .b: %tuple.type.05f} [concrete = constants.%struct_type.a.b.120]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %struct_type.a.b.120 = ref_binding v, %v.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.2dd605.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.3(%self.param: ref %tuple.type.05f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18_3.4(%self.param: ref %struct_type.a.b.120) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
