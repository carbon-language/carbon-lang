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
  var ints: [i32; 6] = (
    8,
    9,
    0x8,
    0b1000,
    2147483647,
    0x7FFFFFFF,
  );
  var floats: [f64; 6] = (
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
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %array_type.a36: type = array_type %int_6, %i32 [template]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %int_2147483647.d89: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %tuple.type.27c: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.29e: <bound method> = bound_method %int_8.b85, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5a6: <specific function> = specific_function %Convert.bound.29e, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_8.f03: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.cc8: <bound method> = bound_method %int_9.988, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b8b: <specific function> = specific_function %Convert.bound.cc8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.82c: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2147483647.d89, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.a96: <specific function> = specific_function %Convert.bound.ef9, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2ba: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array.11d: %array_type.a36 = tuple_value (%int_8.f03, %int_9.82c, %int_8.f03, %int_8.f03, %int_2147483647.2ba, %int_2147483647.2ba) [template]
// CHECK:STDOUT:   %array_type.72b: type = array_type %int_6, f64 [template]
// CHECK:STDOUT:   %float.952: f64 = float_literal 0.90000000000000002 [template]
// CHECK:STDOUT:   %float.298: f64 = float_literal 8 [template]
// CHECK:STDOUT:   %float.dcb: f64 = float_literal 80 [template]
// CHECK:STDOUT:   %float.1d0: f64 = float_literal 1.0E+7 [template]
// CHECK:STDOUT:   %float.9f6: f64 = float_literal 1.0E+8 [template]
// CHECK:STDOUT:   %float.401: f64 = float_literal 1.0E-8 [template]
// CHECK:STDOUT:   %tuple.type.635: type = tuple_type (f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %array.a2f: %array_type.72b = tuple_value (%float.952, %float.298, %float.dcb, %float.1d0, %float.9f6, %float.401) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     .Float = %import_ref.20bd
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ints.var: ref %array_type.a36 = var ints
// CHECK:STDOUT:   %ints: ref %array_type.a36 = bind_name ints, %ints.var
// CHECK:STDOUT:   %int_8.loc15: Core.IntLiteral = int_value 8 [template = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.988]
// CHECK:STDOUT:   %int_8.loc17: Core.IntLiteral = int_value 8 [template = constants.%int_8.b85]
// CHECK:STDOUT:   %int_8.loc18: Core.IntLiteral = int_value 8 [template = constants.%int_8.b85]
// CHECK:STDOUT:   %int_2147483647.loc19: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %int_2147483647.loc20: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %.loc21_3.1: %tuple.type.27c = tuple_literal (%int_8.loc15, %int_9, %int_8.loc17, %int_8.loc18, %int_2147483647.loc19, %int_2147483647.loc20)
// CHECK:STDOUT:   %impl.elem0.loc21_3.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc21_3.1: <bound method> = bound_method %int_8.loc15, %impl.elem0.loc21_3.1 [template = constants.%Convert.bound.29e]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.1: <specific function> = specific_function %Convert.bound.loc21_3.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5a6]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.1: init %i32 = call %Convert.specific_fn.loc21_3.1(%int_8.loc15) [template = constants.%int_8.f03]
// CHECK:STDOUT:   %.loc21_3.2: init %i32 = converted %int_8.loc15, %int.convert_checked.loc21_3.1 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc21_3.3: ref %i32 = array_index %ints.var, %int_0.loc21
// CHECK:STDOUT:   %.loc21_3.4: init %i32 = initialize_from %.loc21_3.2 to %.loc21_3.3 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %impl.elem0.loc21_3.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc21_3.2: <bound method> = bound_method %int_9, %impl.elem0.loc21_3.2 [template = constants.%Convert.bound.cc8]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.2: <specific function> = specific_function %Convert.bound.loc21_3.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b8b]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.2: init %i32 = call %Convert.specific_fn.loc21_3.2(%int_9) [template = constants.%int_9.82c]
// CHECK:STDOUT:   %.loc21_3.5: init %i32 = converted %int_9, %int.convert_checked.loc21_3.2 [template = constants.%int_9.82c]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc21_3.6: ref %i32 = array_index %ints.var, %int_1.loc21
// CHECK:STDOUT:   %.loc21_3.7: init %i32 = initialize_from %.loc21_3.5 to %.loc21_3.6 [template = constants.%int_9.82c]
// CHECK:STDOUT:   %impl.elem0.loc21_3.3: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc21_3.3: <bound method> = bound_method %int_8.loc17, %impl.elem0.loc21_3.3 [template = constants.%Convert.bound.29e]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.3: <specific function> = specific_function %Convert.bound.loc21_3.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5a6]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.3: init %i32 = call %Convert.specific_fn.loc21_3.3(%int_8.loc17) [template = constants.%int_8.f03]
// CHECK:STDOUT:   %.loc21_3.8: init %i32 = converted %int_8.loc17, %int.convert_checked.loc21_3.3 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %int_2.loc21: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc21_3.9: ref %i32 = array_index %ints.var, %int_2.loc21
// CHECK:STDOUT:   %.loc21_3.10: init %i32 = initialize_from %.loc21_3.8 to %.loc21_3.9 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %impl.elem0.loc21_3.4: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc21_3.4: <bound method> = bound_method %int_8.loc18, %impl.elem0.loc21_3.4 [template = constants.%Convert.bound.29e]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.4: <specific function> = specific_function %Convert.bound.loc21_3.4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5a6]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.4: init %i32 = call %Convert.specific_fn.loc21_3.4(%int_8.loc18) [template = constants.%int_8.f03]
// CHECK:STDOUT:   %.loc21_3.11: init %i32 = converted %int_8.loc18, %int.convert_checked.loc21_3.4 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %int_3.loc21: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc21_3.12: ref %i32 = array_index %ints.var, %int_3.loc21
// CHECK:STDOUT:   %.loc21_3.13: init %i32 = initialize_from %.loc21_3.11 to %.loc21_3.12 [template = constants.%int_8.f03]
// CHECK:STDOUT:   %impl.elem0.loc21_3.5: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc21_3.5: <bound method> = bound_method %int_2147483647.loc19, %impl.elem0.loc21_3.5 [template = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.5: <specific function> = specific_function %Convert.bound.loc21_3.5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.a96]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.5: init %i32 = call %Convert.specific_fn.loc21_3.5(%int_2147483647.loc19) [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %.loc21_3.14: init %i32 = converted %int_2147483647.loc19, %int.convert_checked.loc21_3.5 [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc21_3.15: ref %i32 = array_index %ints.var, %int_4.loc21
// CHECK:STDOUT:   %.loc21_3.16: init %i32 = initialize_from %.loc21_3.14 to %.loc21_3.15 [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %impl.elem0.loc21_3.6: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc21_3.6: <bound method> = bound_method %int_2147483647.loc20, %impl.elem0.loc21_3.6 [template = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.6: <specific function> = specific_function %Convert.bound.loc21_3.6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.a96]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.6: init %i32 = call %Convert.specific_fn.loc21_3.6(%int_2147483647.loc20) [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %.loc21_3.17: init %i32 = converted %int_2147483647.loc20, %int.convert_checked.loc21_3.6 [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %int_5.loc21: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %.loc21_3.18: ref %i32 = array_index %ints.var, %int_5.loc21
// CHECK:STDOUT:   %.loc21_3.19: init %i32 = initialize_from %.loc21_3.17 to %.loc21_3.18 [template = constants.%int_2147483647.2ba]
// CHECK:STDOUT:   %.loc21_3.20: init %array_type.a36 = array_init (%.loc21_3.4, %.loc21_3.7, %.loc21_3.10, %.loc21_3.13, %.loc21_3.16, %.loc21_3.19) to %ints.var [template = constants.%array.11d]
// CHECK:STDOUT:   %.loc21_4: init %array_type.a36 = converted %.loc21_3.1, %.loc21_3.20 [template = constants.%array.11d]
// CHECK:STDOUT:   assign %ints.var, %.loc21_4
// CHECK:STDOUT:   %floats.var: ref %array_type.72b = var floats
// CHECK:STDOUT:   %floats: ref %array_type.72b = bind_name floats, %floats.var
// CHECK:STDOUT:   %float.loc23: f64 = float_literal 0.90000000000000002 [template = constants.%float.952]
// CHECK:STDOUT:   %float.loc24: f64 = float_literal 8 [template = constants.%float.298]
// CHECK:STDOUT:   %float.loc25: f64 = float_literal 80 [template = constants.%float.dcb]
// CHECK:STDOUT:   %float.loc26: f64 = float_literal 1.0E+7 [template = constants.%float.1d0]
// CHECK:STDOUT:   %float.loc27: f64 = float_literal 1.0E+8 [template = constants.%float.9f6]
// CHECK:STDOUT:   %float.loc28: f64 = float_literal 1.0E-8 [template = constants.%float.401]
// CHECK:STDOUT:   %.loc29_3.1: %tuple.type.635 = tuple_literal (%float.loc23, %float.loc24, %float.loc25, %float.loc26, %float.loc27, %float.loc28)
// CHECK:STDOUT:   %int_0.loc29: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc29_3.2: ref f64 = array_index %floats.var, %int_0.loc29
// CHECK:STDOUT:   %.loc29_3.3: init f64 = initialize_from %float.loc23 to %.loc29_3.2 [template = constants.%float.952]
// CHECK:STDOUT:   %int_1.loc29: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc29_3.4: ref f64 = array_index %floats.var, %int_1.loc29
// CHECK:STDOUT:   %.loc29_3.5: init f64 = initialize_from %float.loc24 to %.loc29_3.4 [template = constants.%float.298]
// CHECK:STDOUT:   %int_2.loc29: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc29_3.6: ref f64 = array_index %floats.var, %int_2.loc29
// CHECK:STDOUT:   %.loc29_3.7: init f64 = initialize_from %float.loc25 to %.loc29_3.6 [template = constants.%float.dcb]
// CHECK:STDOUT:   %int_3.loc29: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc29_3.8: ref f64 = array_index %floats.var, %int_3.loc29
// CHECK:STDOUT:   %.loc29_3.9: init f64 = initialize_from %float.loc26 to %.loc29_3.8 [template = constants.%float.1d0]
// CHECK:STDOUT:   %int_4.loc29: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc29_3.10: ref f64 = array_index %floats.var, %int_4.loc29
// CHECK:STDOUT:   %.loc29_3.11: init f64 = initialize_from %float.loc27 to %.loc29_3.10 [template = constants.%float.9f6]
// CHECK:STDOUT:   %int_5.loc29: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %.loc29_3.12: ref f64 = array_index %floats.var, %int_5.loc29
// CHECK:STDOUT:   %.loc29_3.13: init f64 = initialize_from %float.loc28 to %.loc29_3.12 [template = constants.%float.401]
// CHECK:STDOUT:   %.loc29_3.14: init %array_type.72b = array_init (%.loc29_3.3, %.loc29_3.5, %.loc29_3.7, %.loc29_3.9, %.loc29_3.11, %.loc29_3.13) to %floats.var [template = constants.%array.a2f]
// CHECK:STDOUT:   %.loc29_4: init %array_type.72b = converted %.loc29_3.1, %.loc29_3.14 [template = constants.%array.a2f]
// CHECK:STDOUT:   assign %floats.var, %.loc29_4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
