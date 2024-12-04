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
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_6, %i32 [template]
// CHECK:STDOUT:   %int_8.1: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %int_9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_8.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_8.2: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_9.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2147483647.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array.1: %array_type.1 = tuple_value (%int_8.2, %int_9.2, %int_8.2, %int_8.2, %int_2147483647.2, %int_2147483647.2) [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_6, f64 [template]
// CHECK:STDOUT:   %float.1: f64 = float_literal 0.90000000000000002 [template]
// CHECK:STDOUT:   %float.2: f64 = float_literal 8 [template]
// CHECK:STDOUT:   %float.3: f64 = float_literal 80 [template]
// CHECK:STDOUT:   %float.4: f64 = float_literal 1.0E+7 [template]
// CHECK:STDOUT:   %float.5: f64 = float_literal 1.0E+8 [template]
// CHECK:STDOUT:   %float.6: f64 = float_literal 1.0E-8 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %array.2: %array_type.2 = tuple_value (%float.1, %float.2, %float.3, %float.4, %float.5, %float.6) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.38
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_6.loc14: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %.loc14_14.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_14.2: type = converted %int.make_type_signed, %.loc14_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc14: type = array_type %int_6.loc14, %i32 [template = constants.%array_type.1]
// CHECK:STDOUT:   %ints.var: ref %array_type.1 = var ints
// CHECK:STDOUT:   %ints: ref %array_type.1 = bind_name ints, %ints.var
// CHECK:STDOUT:   %int_8.loc15: Core.IntLiteral = int_value 8 [template = constants.%int_8.1]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.1]
// CHECK:STDOUT:   %int_8.loc17: Core.IntLiteral = int_value 8 [template = constants.%int_8.1]
// CHECK:STDOUT:   %int_8.loc18: Core.IntLiteral = int_value 8 [template = constants.%int_8.1]
// CHECK:STDOUT:   %int_2147483647.loc19: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %int_2147483647.loc20: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %.loc21_3.1: %tuple.type.1 = tuple_literal (%int_8.loc15, %int_9, %int_8.loc17, %int_8.loc18, %int_2147483647.loc19, %int_2147483647.loc20)
// CHECK:STDOUT:   %impl.elem0.loc21_3.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.1: <bound method> = bound_method %int_8.loc15, %impl.elem0.loc21_3.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.1: <specific function> = specific_function %Convert.bound.loc21_3.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.1: init %i32 = call %Convert.specific_fn.loc21_3.1(%int_8.loc15) [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc21_3.2: init %i32 = converted %int_8.loc15, %int.convert_checked.loc21_3.1 [template = constants.%int_8.2]
// CHECK:STDOUT:   %int_0.loc21: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc21_3.3: ref %i32 = array_index %ints.var, %int_0.loc21
// CHECK:STDOUT:   %.loc21_3.4: init %i32 = initialize_from %.loc21_3.2 to %.loc21_3.3 [template = constants.%int_8.2]
// CHECK:STDOUT:   %impl.elem0.loc21_3.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.2: <bound method> = bound_method %int_9, %impl.elem0.loc21_3.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.2: <specific function> = specific_function %Convert.bound.loc21_3.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.2: init %i32 = call %Convert.specific_fn.loc21_3.2(%int_9) [template = constants.%int_9.2]
// CHECK:STDOUT:   %.loc21_3.5: init %i32 = converted %int_9, %int.convert_checked.loc21_3.2 [template = constants.%int_9.2]
// CHECK:STDOUT:   %int_1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc21_3.6: ref %i32 = array_index %ints.var, %int_1.loc21
// CHECK:STDOUT:   %.loc21_3.7: init %i32 = initialize_from %.loc21_3.5 to %.loc21_3.6 [template = constants.%int_9.2]
// CHECK:STDOUT:   %impl.elem0.loc21_3.3: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.3: <bound method> = bound_method %int_8.loc17, %impl.elem0.loc21_3.3 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.3: <specific function> = specific_function %Convert.bound.loc21_3.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.3: init %i32 = call %Convert.specific_fn.loc21_3.3(%int_8.loc17) [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc21_3.8: init %i32 = converted %int_8.loc17, %int.convert_checked.loc21_3.3 [template = constants.%int_8.2]
// CHECK:STDOUT:   %int_2.loc21: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc21_3.9: ref %i32 = array_index %ints.var, %int_2.loc21
// CHECK:STDOUT:   %.loc21_3.10: init %i32 = initialize_from %.loc21_3.8 to %.loc21_3.9 [template = constants.%int_8.2]
// CHECK:STDOUT:   %impl.elem0.loc21_3.4: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.4: <bound method> = bound_method %int_8.loc18, %impl.elem0.loc21_3.4 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.4: <specific function> = specific_function %Convert.bound.loc21_3.4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.4: init %i32 = call %Convert.specific_fn.loc21_3.4(%int_8.loc18) [template = constants.%int_8.2]
// CHECK:STDOUT:   %.loc21_3.11: init %i32 = converted %int_8.loc18, %int.convert_checked.loc21_3.4 [template = constants.%int_8.2]
// CHECK:STDOUT:   %int_3.loc21: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc21_3.12: ref %i32 = array_index %ints.var, %int_3.loc21
// CHECK:STDOUT:   %.loc21_3.13: init %i32 = initialize_from %.loc21_3.11 to %.loc21_3.12 [template = constants.%int_8.2]
// CHECK:STDOUT:   %impl.elem0.loc21_3.5: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.5: <bound method> = bound_method %int_2147483647.loc19, %impl.elem0.loc21_3.5 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.5: <specific function> = specific_function %Convert.bound.loc21_3.5, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.5: init %i32 = call %Convert.specific_fn.loc21_3.5(%int_2147483647.loc19) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc21_3.14: init %i32 = converted %int_2147483647.loc19, %int.convert_checked.loc21_3.5 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc21_3.15: ref %i32 = array_index %ints.var, %int_4.loc21
// CHECK:STDOUT:   %.loc21_3.16: init %i32 = initialize_from %.loc21_3.14 to %.loc21_3.15 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %impl.elem0.loc21_3.6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.6: <bound method> = bound_method %int_2147483647.loc20, %impl.elem0.loc21_3.6 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.6: <specific function> = specific_function %Convert.bound.loc21_3.6, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.6: init %i32 = call %Convert.specific_fn.loc21_3.6(%int_2147483647.loc20) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc21_3.17: init %i32 = converted %int_2147483647.loc20, %int.convert_checked.loc21_3.6 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %int_5.loc21: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %.loc21_3.18: ref %i32 = array_index %ints.var, %int_5.loc21
// CHECK:STDOUT:   %.loc21_3.19: init %i32 = initialize_from %.loc21_3.17 to %.loc21_3.18 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc21_3.20: init %array_type.1 = array_init (%.loc21_3.4, %.loc21_3.7, %.loc21_3.10, %.loc21_3.13, %.loc21_3.16, %.loc21_3.19) to %ints.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc21_4: init %array_type.1 = converted %.loc21_3.1, %.loc21_3.20 [template = constants.%array.1]
// CHECK:STDOUT:   assign %ints.var, %.loc21_4
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int_64) [template = f64]
// CHECK:STDOUT:   %int_6.loc22: Core.IntLiteral = int_value 6 [template = constants.%int_6]
// CHECK:STDOUT:   %.loc22_16.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc22_16.2: type = converted %float.make_type, %.loc22_16.1 [template = f64]
// CHECK:STDOUT:   %array_type.loc22: type = array_type %int_6.loc22, f64 [template = constants.%array_type.2]
// CHECK:STDOUT:   %floats.var: ref %array_type.2 = var floats
// CHECK:STDOUT:   %floats: ref %array_type.2 = bind_name floats, %floats.var
// CHECK:STDOUT:   %float.loc23: f64 = float_literal 0.90000000000000002 [template = constants.%float.1]
// CHECK:STDOUT:   %float.loc24: f64 = float_literal 8 [template = constants.%float.2]
// CHECK:STDOUT:   %float.loc25: f64 = float_literal 80 [template = constants.%float.3]
// CHECK:STDOUT:   %float.loc26: f64 = float_literal 1.0E+7 [template = constants.%float.4]
// CHECK:STDOUT:   %float.loc27: f64 = float_literal 1.0E+8 [template = constants.%float.5]
// CHECK:STDOUT:   %float.loc28: f64 = float_literal 1.0E-8 [template = constants.%float.6]
// CHECK:STDOUT:   %.loc29_3.1: %tuple.type.2 = tuple_literal (%float.loc23, %float.loc24, %float.loc25, %float.loc26, %float.loc27, %float.loc28)
// CHECK:STDOUT:   %int_0.loc29: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc29_3.2: ref f64 = array_index %floats.var, %int_0.loc29
// CHECK:STDOUT:   %.loc29_3.3: init f64 = initialize_from %float.loc23 to %.loc29_3.2 [template = constants.%float.1]
// CHECK:STDOUT:   %int_1.loc29: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc29_3.4: ref f64 = array_index %floats.var, %int_1.loc29
// CHECK:STDOUT:   %.loc29_3.5: init f64 = initialize_from %float.loc24 to %.loc29_3.4 [template = constants.%float.2]
// CHECK:STDOUT:   %int_2.loc29: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc29_3.6: ref f64 = array_index %floats.var, %int_2.loc29
// CHECK:STDOUT:   %.loc29_3.7: init f64 = initialize_from %float.loc25 to %.loc29_3.6 [template = constants.%float.3]
// CHECK:STDOUT:   %int_3.loc29: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc29_3.8: ref f64 = array_index %floats.var, %int_3.loc29
// CHECK:STDOUT:   %.loc29_3.9: init f64 = initialize_from %float.loc26 to %.loc29_3.8 [template = constants.%float.4]
// CHECK:STDOUT:   %int_4.loc29: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc29_3.10: ref f64 = array_index %floats.var, %int_4.loc29
// CHECK:STDOUT:   %.loc29_3.11: init f64 = initialize_from %float.loc27 to %.loc29_3.10 [template = constants.%float.5]
// CHECK:STDOUT:   %int_5.loc29: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %.loc29_3.12: ref f64 = array_index %floats.var, %int_5.loc29
// CHECK:STDOUT:   %.loc29_3.13: init f64 = initialize_from %float.loc28 to %.loc29_3.12 [template = constants.%float.6]
// CHECK:STDOUT:   %.loc29_3.14: init %array_type.2 = array_init (%.loc29_3.3, %.loc29_3.5, %.loc29_3.7, %.loc29_3.9, %.loc29_3.11, %.loc29_3.13) to %floats.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc29_4: init %array_type.2 = converted %.loc29_3.1, %.loc29_3.14 [template = constants.%array.2]
// CHECK:STDOUT:   assign %floats.var, %.loc29_4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
