// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/reorder_fields.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/reorder_fields.carbon

fn MakeI32() -> i32;
fn MakeF64() -> f64;

fn F() -> {.a: i32, .b: f64} {
  let x: {.a: i32, .b: f64} = {.b = MakeF64(), .a = MakeI32()};
  let y: {.b: f64, .a: i32} = x;
  return y;
}

// CHECK:STDOUT: --- reorder_fields.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %MakeI32.type: type = fn_type @MakeI32 [concrete]
// CHECK:STDOUT:   %MakeI32: %MakeI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %MakeF64.type: type = fn_type @MakeF64 [concrete]
// CHECK:STDOUT:   %MakeF64: %MakeF64.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %f64.d77} [concrete]
// CHECK:STDOUT:   %pattern_type.a5c: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b.a: type = struct_type {.b: %f64.d77, .a: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.b43: type = pattern_type %struct_type.b.a [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f0b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.af5: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.87e: %Int.as.Copy.impl.Op.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.26d: %Copy.type = facet_value %i32, (%Copy.impl_witness.f0b) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.26d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.87e, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.a25: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.623: %Float.as.Copy.impl.Op.type.a25 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.160: <witness> = impl_witness imports.%Copy.impl_witness_table.e1b, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.256: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.47e: %Float.as.Copy.impl.Op.type.256 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.2da: %Copy.type = facet_value %f64.d77, (%Copy.impl_witness.160) [concrete]
// CHECK:STDOUT:   %.2ef: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.2da [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.47e, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.419: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.a25) = import_ref Core//prelude/parts/float, loc16_31, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.623)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e1b = impl_witness_table (%Core.import_ref.419), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .MakeI32 = %MakeI32.decl
// CHECK:STDOUT:     .MakeF64 = %MakeF64.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %MakeI32.decl: %MakeI32.type = fn_decl @MakeI32 [concrete = constants.%MakeI32] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeF64.decl: %MakeF64.type = fn_decl @MakeF64 [concrete = constants.%MakeF64] {
// CHECK:STDOUT:     %return.patt: %pattern_type.0ae = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0ae = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:     %return.param: ref %f64.d77 = out_param call_param0
// CHECK:STDOUT:     %return: ref %f64.d77 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a5c = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a5c = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_64.loc18: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc18: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:     %struct_type.a.b.loc18: type = struct_type {.a: %i32, .b: %f64.d77} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:     %return.param: ref %struct_type.a.b = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.a.b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeI32() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeF64() -> %f64.d77;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return.param: %struct_type.a.b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.a5c = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeF64.ref: %MakeF64.type = name_ref MakeF64, file.%MakeF64.decl [concrete = constants.%MakeF64]
// CHECK:STDOUT:   %MakeF64.call: init %f64.d77 = call %MakeF64.ref()
// CHECK:STDOUT:   %MakeI32.ref: %MakeI32.type = name_ref MakeI32, file.%MakeI32.decl [concrete = constants.%MakeI32]
// CHECK:STDOUT:   %MakeI32.call: init %i32 = call %MakeI32.ref()
// CHECK:STDOUT:   %.loc19_62.1: %struct_type.b.a = struct_literal (%MakeF64.call, %MakeI32.call)
// CHECK:STDOUT:   %.loc19_27: type = splice_block %struct_type.a.b.loc19 [concrete = constants.%struct_type.a.b] {
// CHECK:STDOUT:     %int_32.loc19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_64.loc19: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc19: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:     %struct_type.a.b.loc19: type = struct_type {.a: %i32, .b: %f64.d77} [concrete = constants.%struct_type.a.b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19_62.2: %i32 = value_of_initializer %MakeI32.call
// CHECK:STDOUT:   %.loc19_62.3: %i32 = converted %MakeI32.call, %.loc19_62.2
// CHECK:STDOUT:   %.loc19_62.4: %f64.d77 = value_of_initializer %MakeF64.call
// CHECK:STDOUT:   %.loc19_62.5: %f64.d77 = converted %MakeF64.call, %.loc19_62.4
// CHECK:STDOUT:   %struct.loc19: %struct_type.a.b = struct_value (%.loc19_62.3, %.loc19_62.5)
// CHECK:STDOUT:   %.loc19_62.6: %struct_type.a.b = converted %.loc19_62.1, %struct.loc19
// CHECK:STDOUT:   %x: %struct_type.a.b = bind_name x, %.loc19_62.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.b43 = binding_pattern y [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.ref: %struct_type.a.b = name_ref x, %x
// CHECK:STDOUT:   %.loc20_27: type = splice_block %struct_type.b.a [concrete = constants.%struct_type.b.a] {
// CHECK:STDOUT:     %int_64.loc20: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc20: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:     %int_32.loc20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.b.a: type = struct_type {.b: %f64.d77, .a: %i32} [concrete = constants.%struct_type.b.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc20_31.1: %f64.d77 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc20_31.2: %i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %struct.loc20: %struct_type.b.a = struct_value (%.loc20_31.1, %.loc20_31.2)
// CHECK:STDOUT:   %.loc20_31.3: %struct_type.b.a = converted %x.ref, %struct.loc20
// CHECK:STDOUT:   %y: %struct_type.b.a = bind_name y, %.loc20_31.3
// CHECK:STDOUT:   %y.ref: %struct_type.b.a = name_ref y, %y
// CHECK:STDOUT:   %.loc21_10.1: %i32 = struct_access %y.ref, element1
// CHECK:STDOUT:   %impl.elem0.loc21_10.1: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc21_10.1: <bound method> = bound_method %.loc21_10.1, %impl.elem0.loc21_10.1
// CHECK:STDOUT:   %specific_fn.loc21_10.1: <specific function> = specific_function %impl.elem0.loc21_10.1, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_10.2: <bound method> = bound_method %.loc21_10.1, %specific_fn.loc21_10.1
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc21_10.2(%.loc21_10.1)
// CHECK:STDOUT:   %.loc21_10.2: ref %i32 = struct_access %return, element1
// CHECK:STDOUT:   %.loc21_10.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call to %.loc21_10.2
// CHECK:STDOUT:   %.loc21_10.4: %f64.d77 = struct_access %y.ref, element0
// CHECK:STDOUT:   %impl.elem0.loc21_10.2: %.2ef = impl_witness_access constants.%Copy.impl_witness.160, element0 [concrete = constants.%Float.as.Copy.impl.Op.47e]
// CHECK:STDOUT:   %bound_method.loc21_10.3: <bound method> = bound_method %.loc21_10.4, %impl.elem0.loc21_10.2
// CHECK:STDOUT:   %specific_fn.loc21_10.2: <specific function> = specific_function %impl.elem0.loc21_10.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_10.4: <bound method> = bound_method %.loc21_10.4, %specific_fn.loc21_10.2
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.d77 = call %bound_method.loc21_10.4(%.loc21_10.4)
// CHECK:STDOUT:   %.loc21_10.5: ref %f64.d77 = struct_access %return, element0
// CHECK:STDOUT:   %.loc21_10.6: init %f64.d77 = initialize_from %Float.as.Copy.impl.Op.call to %.loc21_10.5
// CHECK:STDOUT:   %.loc21_10.7: init %struct_type.a.b = struct_init (%.loc21_10.3, %.loc21_10.6) to %return
// CHECK:STDOUT:   %.loc21_11: init %struct_type.a.b = converted %y.ref, %.loc21_10.7
// CHECK:STDOUT:   return %.loc21_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
