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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.6: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.6, %i32 [template]
// CHECK:STDOUT:   %int.8.1: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %int.9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %int.2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.8.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.8.2: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.9.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %int.4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int.2147483647.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2147483647.2: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %int.5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array.1: %.1 = tuple_value (%int.8.2, %int.9.2, %int.8.2, %int.8.2, %int.2147483647.2, %int.2147483647.2) [template]
// CHECK:STDOUT:   %int.64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = array_type %int.6, f64 [template]
// CHECK:STDOUT:   %.9: f64 = float_literal 0.90000000000000002 [template]
// CHECK:STDOUT:   %.10: f64 = float_literal 8 [template]
// CHECK:STDOUT:   %.11: f64 = float_literal 80 [template]
// CHECK:STDOUT:   %.12: f64 = float_literal 1.0E+7 [template]
// CHECK:STDOUT:   %.13: f64 = float_literal 1.0E+8 [template]
// CHECK:STDOUT:   %.14: f64 = float_literal 1.0E-8 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.7 = tuple_value (%.9, %.10, %.11, %.12, %.13, %.14) [template]
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %int.6.loc14: Core.IntLiteral = int_value 6 [template = constants.%int.6]
// CHECK:STDOUT:   %.loc14_14.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_14.2: type = converted %int.make_type_signed, %.loc14_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_20: type = array_type %int.6.loc14, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %ints.var: ref %.1 = var ints
// CHECK:STDOUT:   %ints: ref %.1 = bind_name ints, %ints.var
// CHECK:STDOUT:   %int.8.loc15: Core.IntLiteral = int_value 8 [template = constants.%int.8.1]
// CHECK:STDOUT:   %int.9: Core.IntLiteral = int_value 9 [template = constants.%int.9.1]
// CHECK:STDOUT:   %int.8.loc17: Core.IntLiteral = int_value 8 [template = constants.%int.8.1]
// CHECK:STDOUT:   %int.8.loc18: Core.IntLiteral = int_value 8 [template = constants.%int.8.1]
// CHECK:STDOUT:   %int.2147483647.loc19: Core.IntLiteral = int_value 2147483647 [template = constants.%int.2147483647.1]
// CHECK:STDOUT:   %int.2147483647.loc20: Core.IntLiteral = int_value 2147483647 [template = constants.%int.2147483647.1]
// CHECK:STDOUT:   %.loc21_3.1: %tuple.type.1 = tuple_literal (%int.8.loc15, %int.9, %int.8.loc17, %int.8.loc18, %int.2147483647.loc19, %int.2147483647.loc20)
// CHECK:STDOUT:   %.loc21_3.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.1: <bound method> = bound_method %int.8.loc15, %.loc21_3.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.1: <specific function> = specific_function %Convert.bound.loc21_3.1, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.1: init %i32 = call %Convert.specific_fn.loc21_3.1(%int.8.loc15) [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc21_3.3: init %i32 = converted %int.8.loc15, %int.convert_checked.loc21_3.1 [template = constants.%int.8.2]
// CHECK:STDOUT:   %int.0.loc21: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc21_3.4: ref %i32 = array_index %ints.var, %int.0.loc21
// CHECK:STDOUT:   %.loc21_3.5: init %i32 = initialize_from %.loc21_3.3 to %.loc21_3.4 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc21_3.6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.2: <bound method> = bound_method %int.9, %.loc21_3.6 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.2: <specific function> = specific_function %Convert.bound.loc21_3.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.2: init %i32 = call %Convert.specific_fn.loc21_3.2(%int.9) [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc21_3.7: init %i32 = converted %int.9, %int.convert_checked.loc21_3.2 [template = constants.%int.9.2]
// CHECK:STDOUT:   %int.1.loc21: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc21_3.8: ref %i32 = array_index %ints.var, %int.1.loc21
// CHECK:STDOUT:   %.loc21_3.9: init %i32 = initialize_from %.loc21_3.7 to %.loc21_3.8 [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc21_3.10: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.3: <bound method> = bound_method %int.8.loc17, %.loc21_3.10 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.3: <specific function> = specific_function %Convert.bound.loc21_3.3, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.3: init %i32 = call %Convert.specific_fn.loc21_3.3(%int.8.loc17) [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc21_3.11: init %i32 = converted %int.8.loc17, %int.convert_checked.loc21_3.3 [template = constants.%int.8.2]
// CHECK:STDOUT:   %int.2.loc21: Core.IntLiteral = int_value 2 [template = constants.%int.2]
// CHECK:STDOUT:   %.loc21_3.12: ref %i32 = array_index %ints.var, %int.2.loc21
// CHECK:STDOUT:   %.loc21_3.13: init %i32 = initialize_from %.loc21_3.11 to %.loc21_3.12 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc21_3.14: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.4: <bound method> = bound_method %int.8.loc18, %.loc21_3.14 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.4: <specific function> = specific_function %Convert.bound.loc21_3.4, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.4: init %i32 = call %Convert.specific_fn.loc21_3.4(%int.8.loc18) [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc21_3.15: init %i32 = converted %int.8.loc18, %int.convert_checked.loc21_3.4 [template = constants.%int.8.2]
// CHECK:STDOUT:   %int.3.loc21: Core.IntLiteral = int_value 3 [template = constants.%int.3]
// CHECK:STDOUT:   %.loc21_3.16: ref %i32 = array_index %ints.var, %int.3.loc21
// CHECK:STDOUT:   %.loc21_3.17: init %i32 = initialize_from %.loc21_3.15 to %.loc21_3.16 [template = constants.%int.8.2]
// CHECK:STDOUT:   %.loc21_3.18: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.5: <bound method> = bound_method %int.2147483647.loc19, %.loc21_3.18 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.5: <specific function> = specific_function %Convert.bound.loc21_3.5, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.5: init %i32 = call %Convert.specific_fn.loc21_3.5(%int.2147483647.loc19) [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc21_3.19: init %i32 = converted %int.2147483647.loc19, %int.convert_checked.loc21_3.5 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %int.4.loc21: Core.IntLiteral = int_value 4 [template = constants.%int.4]
// CHECK:STDOUT:   %.loc21_3.20: ref %i32 = array_index %ints.var, %int.4.loc21
// CHECK:STDOUT:   %.loc21_3.21: init %i32 = initialize_from %.loc21_3.19 to %.loc21_3.20 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc21_3.22: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21_3.6: <bound method> = bound_method %int.2147483647.loc20, %.loc21_3.22 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc21_3.6: <specific function> = specific_function %Convert.bound.loc21_3.6, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.6: init %i32 = call %Convert.specific_fn.loc21_3.6(%int.2147483647.loc20) [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc21_3.23: init %i32 = converted %int.2147483647.loc20, %int.convert_checked.loc21_3.6 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %int.5.loc21: Core.IntLiteral = int_value 5 [template = constants.%int.5]
// CHECK:STDOUT:   %.loc21_3.24: ref %i32 = array_index %ints.var, %int.5.loc21
// CHECK:STDOUT:   %.loc21_3.25: init %i32 = initialize_from %.loc21_3.23 to %.loc21_3.24 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc21_3.26: init %.1 = array_init (%.loc21_3.5, %.loc21_3.9, %.loc21_3.13, %.loc21_3.17, %.loc21_3.21, %.loc21_3.25) to %ints.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc21_4: init %.1 = converted %.loc21_3.1, %.loc21_3.26 [template = constants.%array.1]
// CHECK:STDOUT:   assign %ints.var, %.loc21_4
// CHECK:STDOUT:   %int.64: Core.IntLiteral = int_value 64 [template = constants.%int.64]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%int.64) [template = f64]
// CHECK:STDOUT:   %int.6.loc22: Core.IntLiteral = int_value 6 [template = constants.%int.6]
// CHECK:STDOUT:   %.loc22_16.1: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc22_16.2: type = converted %float.make_type, %.loc22_16.1 [template = f64]
// CHECK:STDOUT:   %.loc22_22: type = array_type %int.6.loc22, f64 [template = constants.%.7]
// CHECK:STDOUT:   %floats.var: ref %.7 = var floats
// CHECK:STDOUT:   %floats: ref %.7 = bind_name floats, %floats.var
// CHECK:STDOUT:   %.loc23: f64 = float_literal 0.90000000000000002 [template = constants.%.9]
// CHECK:STDOUT:   %.loc24: f64 = float_literal 8 [template = constants.%.10]
// CHECK:STDOUT:   %.loc25: f64 = float_literal 80 [template = constants.%.11]
// CHECK:STDOUT:   %.loc26: f64 = float_literal 1.0E+7 [template = constants.%.12]
// CHECK:STDOUT:   %.loc27: f64 = float_literal 1.0E+8 [template = constants.%.13]
// CHECK:STDOUT:   %.loc28: f64 = float_literal 1.0E-8 [template = constants.%.14]
// CHECK:STDOUT:   %.loc29_3.1: %tuple.type.2 = tuple_literal (%.loc23, %.loc24, %.loc25, %.loc26, %.loc27, %.loc28)
// CHECK:STDOUT:   %int.0.loc29: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc29_3.2: ref f64 = array_index %floats.var, %int.0.loc29
// CHECK:STDOUT:   %.loc29_3.3: init f64 = initialize_from %.loc23 to %.loc29_3.2 [template = constants.%.9]
// CHECK:STDOUT:   %int.1.loc29: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc29_3.4: ref f64 = array_index %floats.var, %int.1.loc29
// CHECK:STDOUT:   %.loc29_3.5: init f64 = initialize_from %.loc24 to %.loc29_3.4 [template = constants.%.10]
// CHECK:STDOUT:   %int.2.loc29: Core.IntLiteral = int_value 2 [template = constants.%int.2]
// CHECK:STDOUT:   %.loc29_3.6: ref f64 = array_index %floats.var, %int.2.loc29
// CHECK:STDOUT:   %.loc29_3.7: init f64 = initialize_from %.loc25 to %.loc29_3.6 [template = constants.%.11]
// CHECK:STDOUT:   %int.3.loc29: Core.IntLiteral = int_value 3 [template = constants.%int.3]
// CHECK:STDOUT:   %.loc29_3.8: ref f64 = array_index %floats.var, %int.3.loc29
// CHECK:STDOUT:   %.loc29_3.9: init f64 = initialize_from %.loc26 to %.loc29_3.8 [template = constants.%.12]
// CHECK:STDOUT:   %int.4.loc29: Core.IntLiteral = int_value 4 [template = constants.%int.4]
// CHECK:STDOUT:   %.loc29_3.10: ref f64 = array_index %floats.var, %int.4.loc29
// CHECK:STDOUT:   %.loc29_3.11: init f64 = initialize_from %.loc27 to %.loc29_3.10 [template = constants.%.13]
// CHECK:STDOUT:   %int.5.loc29: Core.IntLiteral = int_value 5 [template = constants.%int.5]
// CHECK:STDOUT:   %.loc29_3.12: ref f64 = array_index %floats.var, %int.5.loc29
// CHECK:STDOUT:   %.loc29_3.13: init f64 = initialize_from %.loc28 to %.loc29_3.12 [template = constants.%.14]
// CHECK:STDOUT:   %.loc29_3.14: init %.7 = array_init (%.loc29_3.3, %.loc29_3.5, %.loc29_3.7, %.loc29_3.9, %.loc29_3.11, %.loc29_3.13) to %floats.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc29_4: init %.7 = converted %.loc29_3.1, %.loc29_3.14 [template = constants.%array.2]
// CHECK:STDOUT:   assign %floats.var, %.loc29_4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
