// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/numeric_literals.carbon

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: array(i32, 6) = (
    8,
    9,
    0x8,
    0b1000,
    2147483647,
    0x7FFFFFFF,
  );
  var floats: array(f64, 6) = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8
  );
}

// CHECK:STDOUT: --- numeric_literals.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %array_type.d49: type = array_type %int_6, %i32 [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_2147483647.d89: Core.IntLiteral = int_value 2147483647 [concrete]
// CHECK:STDOUT:   %tuple.type.27c: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.e09: <bound method> = bound_method %int_8.b85, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.02d: <bound method> = bound_method %int_8.b85, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.9e2: <bound method> = bound_method %int_9.988, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.cd3: <bound method> = bound_method %int_9.988, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.f38: <bound method> = bound_method %int_2147483647.d89, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.8e1: <bound method> = bound_method %int_2147483647.d89, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2147483647.a74: %i32 = int_value 2147483647 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %array.ae2: %array_type.d49 = tuple_value (%int_8.98c, %int_9.f88, %int_8.98c, %int_8.98c, %int_2147483647.a74, %int_2147483647.a74) [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.72b: type = array_type %int_6, f64 [concrete]
// CHECK:STDOUT:   %float.952: f64 = float_literal 0.90000000000000002 [concrete]
// CHECK:STDOUT:   %float.298: f64 = float_literal 8 [concrete]
// CHECK:STDOUT:   %float.dcb: f64 = float_literal 80 [concrete]
// CHECK:STDOUT:   %float.1d0: f64 = float_literal 1.0E+7 [concrete]
// CHECK:STDOUT:   %float.9f6: f64 = float_literal 1.0E+8 [concrete]
// CHECK:STDOUT:   %float.401: f64 = float_literal 1.0E-8 [concrete]
// CHECK:STDOUT:   %tuple.type.635: type = tuple_type (f64, f64, f64, f64, f64, f64) [concrete]
// CHECK:STDOUT:   %array.a2f: %array_type.72b = tuple_value (%float.952, %float.298, %float.dcb, %float.1d0, %float.9f6, %float.401) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ints.patt: %array_type.d49 = binding_pattern ints
// CHECK:STDOUT:     %.loc14_3.1: %array_type.d49 = var_pattern %ints.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ints.var: ref %array_type.d49 = var ints
// CHECK:STDOUT:   %int_8.loc15: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %int_8.loc17: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_8.loc18: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_2147483647.loc19: Core.IntLiteral = int_value 2147483647 [concrete = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %int_2147483647.loc20: Core.IntLiteral = int_value 2147483647 [concrete = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %.loc21_3.1: %tuple.type.27c = tuple_literal (%int_8.loc15, %int_9, %int_8.loc17, %int_8.loc18, %int_2147483647.loc19, %int_2147483647.loc20)
// CHECK:STDOUT:   %impl.elem0.loc21_3.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_3.1: <bound method> = bound_method %int_8.loc15, %impl.elem0.loc21_3.1 [concrete = constants.%Convert.bound.e09]
// CHECK:STDOUT:   %specific_fn.loc21_3.1: <specific function> = specific_function %impl.elem0.loc21_3.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_3.2: <bound method> = bound_method %int_8.loc15, %specific_fn.loc21_3.1 [concrete = constants.%bound_method.02d]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.1: init %i32 = call %bound_method.loc21_3.2(%int_8.loc15) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc21_3.2: init %i32 = converted %int_8.loc15, %int.convert_checked.loc21_3.1 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc21_3.3: ref %i32 = array_index %ints.var, %int_0.loc21
// CHECK:STDOUT:   %.loc21_3.4: init %i32 = initialize_from %.loc21_3.2 to %.loc21_3.3 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc21_3.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_3.3: <bound method> = bound_method %int_9, %impl.elem0.loc21_3.2 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc21_3.2: <specific function> = specific_function %impl.elem0.loc21_3.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_3.4: <bound method> = bound_method %int_9, %specific_fn.loc21_3.2 [concrete = constants.%bound_method.cd3]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.2: init %i32 = call %bound_method.loc21_3.4(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc21_3.5: init %i32 = converted %int_9, %int.convert_checked.loc21_3.2 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc21_3.6: ref %i32 = array_index %ints.var, %int_1.loc21
// CHECK:STDOUT:   %.loc21_3.7: init %i32 = initialize_from %.loc21_3.5 to %.loc21_3.6 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %impl.elem0.loc21_3.3: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_3.5: <bound method> = bound_method %int_8.loc17, %impl.elem0.loc21_3.3 [concrete = constants.%Convert.bound.e09]
// CHECK:STDOUT:   %specific_fn.loc21_3.3: <specific function> = specific_function %impl.elem0.loc21_3.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_3.6: <bound method> = bound_method %int_8.loc17, %specific_fn.loc21_3.3 [concrete = constants.%bound_method.02d]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.3: init %i32 = call %bound_method.loc21_3.6(%int_8.loc17) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc21_3.8: init %i32 = converted %int_8.loc17, %int.convert_checked.loc21_3.3 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %int_2.loc21: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc21_3.9: ref %i32 = array_index %ints.var, %int_2.loc21
// CHECK:STDOUT:   %.loc21_3.10: init %i32 = initialize_from %.loc21_3.8 to %.loc21_3.9 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc21_3.4: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_3.7: <bound method> = bound_method %int_8.loc18, %impl.elem0.loc21_3.4 [concrete = constants.%Convert.bound.e09]
// CHECK:STDOUT:   %specific_fn.loc21_3.4: <specific function> = specific_function %impl.elem0.loc21_3.4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_3.8: <bound method> = bound_method %int_8.loc18, %specific_fn.loc21_3.4 [concrete = constants.%bound_method.02d]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.4: init %i32 = call %bound_method.loc21_3.8(%int_8.loc18) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc21_3.11: init %i32 = converted %int_8.loc18, %int.convert_checked.loc21_3.4 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %int_3.loc21: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc21_3.12: ref %i32 = array_index %ints.var, %int_3.loc21
// CHECK:STDOUT:   %.loc21_3.13: init %i32 = initialize_from %.loc21_3.11 to %.loc21_3.12 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc21_3.5: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_3.9: <bound method> = bound_method %int_2147483647.loc19, %impl.elem0.loc21_3.5 [concrete = constants.%Convert.bound.f38]
// CHECK:STDOUT:   %specific_fn.loc21_3.5: <specific function> = specific_function %impl.elem0.loc21_3.5, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_3.10: <bound method> = bound_method %int_2147483647.loc19, %specific_fn.loc21_3.5 [concrete = constants.%bound_method.8e1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.5: init %i32 = call %bound_method.loc21_3.10(%int_2147483647.loc19) [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %.loc21_3.14: init %i32 = converted %int_2147483647.loc19, %int.convert_checked.loc21_3.5 [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc21_3.15: ref %i32 = array_index %ints.var, %int_4.loc21
// CHECK:STDOUT:   %.loc21_3.16: init %i32 = initialize_from %.loc21_3.14 to %.loc21_3.15 [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %impl.elem0.loc21_3.6: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc21_3.11: <bound method> = bound_method %int_2147483647.loc20, %impl.elem0.loc21_3.6 [concrete = constants.%Convert.bound.f38]
// CHECK:STDOUT:   %specific_fn.loc21_3.6: <specific function> = specific_function %impl.elem0.loc21_3.6, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_3.12: <bound method> = bound_method %int_2147483647.loc20, %specific_fn.loc21_3.6 [concrete = constants.%bound_method.8e1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.6: init %i32 = call %bound_method.loc21_3.12(%int_2147483647.loc20) [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %.loc21_3.17: init %i32 = converted %int_2147483647.loc20, %int.convert_checked.loc21_3.6 [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %int_5.loc21: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc21_3.18: ref %i32 = array_index %ints.var, %int_5.loc21
// CHECK:STDOUT:   %.loc21_3.19: init %i32 = initialize_from %.loc21_3.17 to %.loc21_3.18 [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %.loc21_3.20: init %array_type.d49 = array_init (%.loc21_3.4, %.loc21_3.7, %.loc21_3.10, %.loc21_3.13, %.loc21_3.16, %.loc21_3.19) to %ints.var [concrete = constants.%array.ae2]
// CHECK:STDOUT:   %.loc14_3.2: init %array_type.d49 = converted %.loc21_3.1, %.loc21_3.20 [concrete = constants.%array.ae2]
// CHECK:STDOUT:   assign %ints.var, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_25: type = splice_block %array_type.loc14 [concrete = constants.%array_type.d49] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_6.loc14: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:     %array_type.loc14: type = array_type %int_6.loc14, %i32 [concrete = constants.%array_type.d49]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ints: ref %array_type.d49 = bind_name ints, %ints.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %floats.patt: %array_type.72b = binding_pattern floats
// CHECK:STDOUT:     %.loc22_3.1: %array_type.72b = var_pattern %floats.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %floats.var: ref %array_type.72b = var floats
// CHECK:STDOUT:   %float.loc23: f64 = float_literal 0.90000000000000002 [concrete = constants.%float.952]
// CHECK:STDOUT:   %float.loc24: f64 = float_literal 8 [concrete = constants.%float.298]
// CHECK:STDOUT:   %float.loc25: f64 = float_literal 80 [concrete = constants.%float.dcb]
// CHECK:STDOUT:   %float.loc26: f64 = float_literal 1.0E+7 [concrete = constants.%float.1d0]
// CHECK:STDOUT:   %float.loc27: f64 = float_literal 1.0E+8 [concrete = constants.%float.9f6]
// CHECK:STDOUT:   %float.loc28: f64 = float_literal 1.0E-8 [concrete = constants.%float.401]
// CHECK:STDOUT:   %.loc29_3.1: %tuple.type.635 = tuple_literal (%float.loc23, %float.loc24, %float.loc25, %float.loc26, %float.loc27, %float.loc28)
// CHECK:STDOUT:   %int_0.loc29: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc29_3.2: ref f64 = array_index %floats.var, %int_0.loc29
// CHECK:STDOUT:   %.loc29_3.3: init f64 = initialize_from %float.loc23 to %.loc29_3.2 [concrete = constants.%float.952]
// CHECK:STDOUT:   %int_1.loc29: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc29_3.4: ref f64 = array_index %floats.var, %int_1.loc29
// CHECK:STDOUT:   %.loc29_3.5: init f64 = initialize_from %float.loc24 to %.loc29_3.4 [concrete = constants.%float.298]
// CHECK:STDOUT:   %int_2.loc29: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc29_3.6: ref f64 = array_index %floats.var, %int_2.loc29
// CHECK:STDOUT:   %.loc29_3.7: init f64 = initialize_from %float.loc25 to %.loc29_3.6 [concrete = constants.%float.dcb]
// CHECK:STDOUT:   %int_3.loc29: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc29_3.8: ref f64 = array_index %floats.var, %int_3.loc29
// CHECK:STDOUT:   %.loc29_3.9: init f64 = initialize_from %float.loc26 to %.loc29_3.8 [concrete = constants.%float.1d0]
// CHECK:STDOUT:   %int_4.loc29: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc29_3.10: ref f64 = array_index %floats.var, %int_4.loc29
// CHECK:STDOUT:   %.loc29_3.11: init f64 = initialize_from %float.loc27 to %.loc29_3.10 [concrete = constants.%float.9f6]
// CHECK:STDOUT:   %int_5.loc29: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc29_3.12: ref f64 = array_index %floats.var, %int_5.loc29
// CHECK:STDOUT:   %.loc29_3.13: init f64 = initialize_from %float.loc28 to %.loc29_3.12 [concrete = constants.%float.401]
// CHECK:STDOUT:   %.loc29_3.14: init %array_type.72b = array_init (%.loc29_3.3, %.loc29_3.5, %.loc29_3.7, %.loc29_3.9, %.loc29_3.11, %.loc29_3.13) to %floats.var [concrete = constants.%array.a2f]
// CHECK:STDOUT:   %.loc22_3.2: init %array_type.72b = converted %.loc29_3.1, %.loc29_3.14 [concrete = constants.%array.a2f]
// CHECK:STDOUT:   assign %floats.var, %.loc22_3.2
// CHECK:STDOUT:   %.loc22_27: type = splice_block %array_type.loc22 [concrete = constants.%array_type.72b] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %int_6.loc22: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:     %.loc22_21.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc22_21.2: type = converted %float.make_type, %.loc22_21.1 [concrete = f64]
// CHECK:STDOUT:     %array_type.loc22: type = array_type %int_6.loc22, f64 [concrete = constants.%array_type.72b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %floats: ref %array_type.72b = bind_name floats, %floats.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
