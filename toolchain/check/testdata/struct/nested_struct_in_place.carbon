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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.531: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.531 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.9f5: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %.070: Core.Form = init_form %tuple.type.9f5 [concrete]
// CHECK:STDOUT:   %pattern_type.c2f: type = pattern_type %tuple.type.9f5 [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.c2f = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.c2f = return_slot_pattern %return.param_patt, %tuple.type.9f5 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.ce2: type = struct_type {.a: %tuple.type.9f5, .b: %tuple.type.9f5} [concrete]
// CHECK:STDOUT:   %pattern_type.f8f: type = pattern_type %struct_type.a.b.ce2 [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.f8f = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.f8f = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.loc18_3.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc18_3.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.3: type = fn_type @DestroyImpl.WithSelf.Op.loc18_3.3 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.3: %DestroyImpl.WithSelf.Op.type.b77747.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.3: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.3), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %facet_value.911: %facet_type = facet_value %i32, (%custom_witness.2cb555.1, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.0f8: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.911) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.5c6: %facet_type = facet_value %tuple.type.9f5, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.c18: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.5c6) [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.34e: %facet_type = facet_value %struct_type.a.b.ce2, (%custom_witness.2cb555.3, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.aa6: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.34e) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c2f = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.c2f = return_slot_pattern %return.param_patt, %.loc15_25.2 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc15_12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc15_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_25.1: %tuple.type.531 = tuple_literal (%i32.loc15_12, %i32.loc15_17, %i32.loc15_22) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc15_25.2: type = converted %.loc15_25.1, constants.%tuple.type.9f5 [concrete = constants.%tuple.type.9f5]
// CHECK:STDOUT:     %.loc15_25.3: Core.Form = init_form %.loc15_25.2 [concrete = constants.%.070]
// CHECK:STDOUT:     %return.param: ref %tuple.type.9f5 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.9f5 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %tuple.type.9f5;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %struct_type.a.b.ce2 = var_storage %v.var_patt
// CHECK:STDOUT:   %F.ref.loc18_68: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_81.1: ref %tuple.type.9f5 = struct_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc18_70: init %tuple.type.9f5 to %.loc18_81.1 = call %F.ref.loc18_68()
// CHECK:STDOUT:   %F.ref.loc18_78: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_81.2: ref %tuple.type.9f5 = struct_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc18_80: init %tuple.type.9f5 to %.loc18_81.2 = call %F.ref.loc18_78()
// CHECK:STDOUT:   %.loc18_81.3: %struct_type.a.b.ce2 = struct_literal (%F.call.loc18_70, %F.call.loc18_80)
// CHECK:STDOUT:   %.loc18_81.4: init %struct_type.a.b.ce2 to %v.var = struct_init (%F.call.loc18_70, %F.call.loc18_80)
// CHECK:STDOUT:   %.loc18_3.1: init %struct_type.a.b.ce2 = converted %.loc18_81.3, %.loc18_81.4
// CHECK:STDOUT:   assign %v.var, %.loc18_3.1
// CHECK:STDOUT:   %.loc18_58: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.ce2] {
// CHECK:STDOUT:     %i32.loc18_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_33: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.1: %tuple.type.531 = tuple_literal (%i32.loc18_23, %i32.loc18_28, %i32.loc18_33) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc18_36.2: type = converted %.loc18_36.1, constants.%tuple.type.9f5 [concrete = constants.%tuple.type.9f5]
// CHECK:STDOUT:     %i32.loc18_44: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_49: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc18_54: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_57.1: %tuple.type.531 = tuple_literal (%i32.loc18_44, %i32.loc18_49, %i32.loc18_54) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc18_57.2: type = converted %.loc18_57.1, constants.%tuple.type.9f5 [concrete = constants.%tuple.type.9f5]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %tuple.type.9f5, .b: %tuple.type.9f5} [concrete = constants.%struct_type.a.b.ce2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %struct_type.a.b.ce2 = wrapper_binding v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.f8f = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.f8f = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc18_3.1: %facet_type = facet_value constants.%struct_type.a.b.ce2, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34e]
// CHECK:STDOUT:   %.loc18_3.2: %facet_type = converted constants.%struct_type.a.b.ce2, %facet_value.loc18_3.1 [concrete = constants.%facet_value.34e]
// CHECK:STDOUT:   %facet_value.loc18_3.2: %facet_type = facet_value constants.%struct_type.a.b.ce2, (constants.%custom_witness.2cb555.3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.34e]
// CHECK:STDOUT:   %.loc18_3.3: %facet_type = converted constants.%struct_type.a.b.ce2, %facet_value.loc18_3.2 [concrete = constants.%facet_value.34e]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.34e) [concrete = constants.%SelfDestruct.specific_fn.aa6]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%v.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc18_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc18_3.2(%self.param: ref %tuple.type.9f5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc18_3.3(%self.param: ref %struct_type.a.b.ce2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.2(%self.param: ref %tuple.type.9f5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %self.param, element2
// CHECK:STDOUT:   %facet_value.loc18_3.1: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc18_3.1: %facet_type = converted constants.%i32, %facet_value.loc18_3.1 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc18_3.2: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc18_3.2: %facet_type = converted constants.%i32, %facet_value.loc18_3.2 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc18_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc18_3.3: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc18_3.3: %facet_type = converted constants.%i32, %facet_value.loc18_3.3 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc18_3.4: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc18_3.4: %facet_type = converted constants.%i32, %facet_value.loc18_3.4 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc18_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc18_3.5: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc18_3.5: %facet_type = converted constants.%i32, %facet_value.loc18_3.5 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc18_3.6: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc18_3.6: %facet_type = converted constants.%i32, %facet_value.loc18_3.6 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc18_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%tuple.elem0)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18_3.3(%self.param: ref %struct_type.a.b.ce2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_3.1: ref %tuple.type.9f5 = struct_access %self.param, element1
// CHECK:STDOUT:   %facet_value.loc18_3.1: %facet_type = facet_value constants.%tuple.type.9f5, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %.loc18_3.2: %facet_type = converted constants.%tuple.type.9f5, %facet_value.loc18_3.1 [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %facet_value.loc18_3.2: %facet_type = facet_value constants.%tuple.type.9f5, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %.loc18_3.3: %facet_type = converted constants.%tuple.type.9f5, %facet_value.loc18_3.2 [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.5c6) [concrete = constants.%SelfDestruct.specific_fn.c18]
// CHECK:STDOUT:   %SelfDestruct.call.loc18_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc18_3.1)
// CHECK:STDOUT:   %.loc18_3.4: ref %tuple.type.9f5 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc18_3.3: %facet_type = facet_value constants.%tuple.type.9f5, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %.loc18_3.5: %facet_type = converted constants.%tuple.type.9f5, %facet_value.loc18_3.3 [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %facet_value.loc18_3.4: %facet_type = facet_value constants.%tuple.type.9f5, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %.loc18_3.6: %facet_type = converted constants.%tuple.type.9f5, %facet_value.loc18_3.4 [concrete = constants.%facet_value.5c6]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.5c6) [concrete = constants.%SelfDestruct.specific_fn.c18]
// CHECK:STDOUT:   %SelfDestruct.call.loc18_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc18_3.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
