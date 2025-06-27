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
  var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
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
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.b5a: type = pattern_type %tuple.type.189 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b.2f9: type = struct_type {.a: %tuple.type.189, .b: %tuple.type.189} [concrete]
// CHECK:STDOUT:   %pattern_type.3c2: type = pattern_type %struct_type.a.b.2f9 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.3, @impl.49c(%T) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.c0f: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.type.19f: type = fn_type @Op.3, @impl.49c(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Op.672: %Op.type.19f = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.ba8: %Destroy.type = facet_value %tuple.type.189, (%Destroy.impl_witness.c0f) [concrete]
// CHECK:STDOUT:   %.cce: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.ba8 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.cf9: <specific function> = specific_function %Op.672, @Op.3(%tuple.type.189) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.0dd: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%struct_type.a.b.2f9) [concrete]
// CHECK:STDOUT:   %Op.type.840: type = fn_type @Op.3, @impl.49c(%struct_type.a.b.2f9) [concrete]
// CHECK:STDOUT:   %Op.52b: %Op.type.840 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.4f2: %Destroy.type = facet_value %struct_type.a.b.2f9, (%Destroy.impl_witness.0dd) [concrete]
// CHECK:STDOUT:   %.1a1: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.4f2 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.c7e: <specific function> = specific_function %Op.52b, @Op.3(%struct_type.a.b.2f9) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
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
// CHECK:STDOUT:     %return.patt: %pattern_type.b5a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b5a = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc15_12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_25.1: %tuple.type.ff9 = tuple_literal (%i32.loc15_12, %i32.loc15_17, %i32.loc15_22)
// CHECK:STDOUT:     %.loc15_25.2: type = converted %.loc15_25.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %return.param: ref %tuple.type.189 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.189 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %tuple.type.189;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.3c2 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.3c2 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %struct_type.a.b.2f9 = var %v.var_patt
// CHECK:STDOUT:   %F.ref.loc18_61: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_74.1: ref %tuple.type.189 = struct_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc18_63: init %tuple.type.189 = call %F.ref.loc18_61() to %.loc18_74.1
// CHECK:STDOUT:   %F.ref.loc18_71: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc18_74.2: ref %tuple.type.189 = struct_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc18_73: init %tuple.type.189 = call %F.ref.loc18_71() to %.loc18_74.2
// CHECK:STDOUT:   %.loc18_74.3: %struct_type.a.b.2f9 = struct_literal (%F.call.loc18_63, %F.call.loc18_73)
// CHECK:STDOUT:   %.loc18_74.4: init %struct_type.a.b.2f9 = struct_init (%F.call.loc18_63, %F.call.loc18_73) to %v.var
// CHECK:STDOUT:   %.loc18_3.1: init %struct_type.a.b.2f9 = converted %.loc18_74.3, %.loc18_74.4
// CHECK:STDOUT:   assign %v.var, %.loc18_3.1
// CHECK:STDOUT:   %.loc18_51: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.2f9] {
// CHECK:STDOUT:     %int_32.loc18_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_29.1: %tuple.type.ff9 = tuple_literal (%i32.loc18_16, %i32.loc18_21, %i32.loc18_26)
// CHECK:STDOUT:     %.loc18_29.2: type = converted %.loc18_29.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %int_32.loc18_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_42: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_42: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_47: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_47: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_50.1: %tuple.type.ff9 = tuple_literal (%i32.loc18_37, %i32.loc18_42, %i32.loc18_47)
// CHECK:STDOUT:     %.loc18_50.2: type = converted %.loc18_50.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: %tuple.type.189, .b: %tuple.type.189} [concrete = constants.%struct_type.a.b.2f9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %struct_type.a.b.2f9 = bind_name v, %v.var
// CHECK:STDOUT:   %impl.elem0.loc18_74.1: %.cce = impl_witness_access constants.%Destroy.impl_witness.c0f, element0 [concrete = constants.%Op.672]
// CHECK:STDOUT:   %bound_method.loc18_74.1: <bound method> = bound_method %.loc18_74.2, %impl.elem0.loc18_74.1
// CHECK:STDOUT:   %specific_fn.loc18_74.1: <specific function> = specific_function %impl.elem0.loc18_74.1, @Op.3(constants.%tuple.type.189) [concrete = constants.%Op.specific_fn.cf9]
// CHECK:STDOUT:   %bound_method.loc18_74.2: <bound method> = bound_method %.loc18_74.2, %specific_fn.loc18_74.1
// CHECK:STDOUT:   %tuple.elem0.loc18_74.1: ref %i32 = tuple_access %.loc18_74.2, element0
// CHECK:STDOUT:   %.loc18_74.5: %i32 = bind_value %tuple.elem0.loc18_74.1
// CHECK:STDOUT:   %tuple.elem1.loc18_74.1: ref %i32 = tuple_access %.loc18_74.2, element1
// CHECK:STDOUT:   %.loc18_74.6: %i32 = bind_value %tuple.elem1.loc18_74.1
// CHECK:STDOUT:   %tuple.elem2.loc18_74.1: ref %i32 = tuple_access %.loc18_74.2, element2
// CHECK:STDOUT:   %.loc18_74.7: %i32 = bind_value %tuple.elem2.loc18_74.1
// CHECK:STDOUT:   %tuple.loc18_74.1: %tuple.type.189 = tuple_value (%.loc18_74.5, %.loc18_74.6, %.loc18_74.7)
// CHECK:STDOUT:   %.loc18_74.8: %tuple.type.189 = converted %.loc18_74.2, %tuple.loc18_74.1
// CHECK:STDOUT:   %no_op.loc18_74.1: init %empty_tuple.type = call %bound_method.loc18_74.2(%.loc18_74.8)
// CHECK:STDOUT:   %impl.elem0.loc18_74.2: %.cce = impl_witness_access constants.%Destroy.impl_witness.c0f, element0 [concrete = constants.%Op.672]
// CHECK:STDOUT:   %bound_method.loc18_74.3: <bound method> = bound_method %.loc18_74.1, %impl.elem0.loc18_74.2
// CHECK:STDOUT:   %specific_fn.loc18_74.2: <specific function> = specific_function %impl.elem0.loc18_74.2, @Op.3(constants.%tuple.type.189) [concrete = constants.%Op.specific_fn.cf9]
// CHECK:STDOUT:   %bound_method.loc18_74.4: <bound method> = bound_method %.loc18_74.1, %specific_fn.loc18_74.2
// CHECK:STDOUT:   %tuple.elem0.loc18_74.2: ref %i32 = tuple_access %.loc18_74.1, element0
// CHECK:STDOUT:   %.loc18_74.9: %i32 = bind_value %tuple.elem0.loc18_74.2
// CHECK:STDOUT:   %tuple.elem1.loc18_74.2: ref %i32 = tuple_access %.loc18_74.1, element1
// CHECK:STDOUT:   %.loc18_74.10: %i32 = bind_value %tuple.elem1.loc18_74.2
// CHECK:STDOUT:   %tuple.elem2.loc18_74.2: ref %i32 = tuple_access %.loc18_74.1, element2
// CHECK:STDOUT:   %.loc18_74.11: %i32 = bind_value %tuple.elem2.loc18_74.2
// CHECK:STDOUT:   %tuple.loc18_74.2: %tuple.type.189 = tuple_value (%.loc18_74.9, %.loc18_74.10, %.loc18_74.11)
// CHECK:STDOUT:   %.loc18_74.12: %tuple.type.189 = converted %.loc18_74.1, %tuple.loc18_74.2
// CHECK:STDOUT:   %no_op.loc18_74.2: init %empty_tuple.type = call %bound_method.loc18_74.4(%.loc18_74.12)
// CHECK:STDOUT:   %impl.elem0.loc18_3: %.1a1 = impl_witness_access constants.%Destroy.impl_witness.0dd, element0 [concrete = constants.%Op.52b]
// CHECK:STDOUT:   %bound_method.loc18_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc18_3
// CHECK:STDOUT:   %specific_fn.loc18_3: <specific function> = specific_function %impl.elem0.loc18_3, @Op.3(constants.%struct_type.a.b.2f9) [concrete = constants.%Op.specific_fn.c7e]
// CHECK:STDOUT:   %bound_method.loc18_3.2: <bound method> = bound_method %v.var, %specific_fn.loc18_3
// CHECK:STDOUT:   %.loc18_3.2: ref %tuple.type.189 = struct_access %v.var, element0
// CHECK:STDOUT:   %tuple.elem0.loc18_3.1: ref %i32 = tuple_access %.loc18_3.2, element0
// CHECK:STDOUT:   %.loc18_3.3: %i32 = bind_value %tuple.elem0.loc18_3.1
// CHECK:STDOUT:   %tuple.elem1.loc18_3.1: ref %i32 = tuple_access %.loc18_3.2, element1
// CHECK:STDOUT:   %.loc18_3.4: %i32 = bind_value %tuple.elem1.loc18_3.1
// CHECK:STDOUT:   %tuple.elem2.loc18_3.1: ref %i32 = tuple_access %.loc18_3.2, element2
// CHECK:STDOUT:   %.loc18_3.5: %i32 = bind_value %tuple.elem2.loc18_3.1
// CHECK:STDOUT:   %tuple.loc18_3.1: %tuple.type.189 = tuple_value (%.loc18_3.3, %.loc18_3.4, %.loc18_3.5)
// CHECK:STDOUT:   %.loc18_3.6: %tuple.type.189 = converted %.loc18_3.2, %tuple.loc18_3.1
// CHECK:STDOUT:   %.loc18_3.7: ref %tuple.type.189 = struct_access %v.var, element1
// CHECK:STDOUT:   %tuple.elem0.loc18_3.2: ref %i32 = tuple_access %.loc18_3.7, element0
// CHECK:STDOUT:   %.loc18_3.8: %i32 = bind_value %tuple.elem0.loc18_3.2
// CHECK:STDOUT:   %tuple.elem1.loc18_3.2: ref %i32 = tuple_access %.loc18_3.7, element1
// CHECK:STDOUT:   %.loc18_3.9: %i32 = bind_value %tuple.elem1.loc18_3.2
// CHECK:STDOUT:   %tuple.elem2.loc18_3.2: ref %i32 = tuple_access %.loc18_3.7, element2
// CHECK:STDOUT:   %.loc18_3.10: %i32 = bind_value %tuple.elem2.loc18_3.2
// CHECK:STDOUT:   %tuple.loc18_3.2: %tuple.type.189 = tuple_value (%.loc18_3.8, %.loc18_3.9, %.loc18_3.10)
// CHECK:STDOUT:   %.loc18_3.11: %tuple.type.189 = converted %.loc18_3.7, %tuple.loc18_3.2
// CHECK:STDOUT:   %struct: %struct_type.a.b.2f9 = struct_value (%.loc18_3.6, %.loc18_3.11)
// CHECK:STDOUT:   %.loc18_3.12: %struct_type.a.b.2f9 = converted %v.var, %struct
// CHECK:STDOUT:   %no_op.loc18_3: init %empty_tuple.type = call %bound_method.loc18_3.2(%.loc18_3.12)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
