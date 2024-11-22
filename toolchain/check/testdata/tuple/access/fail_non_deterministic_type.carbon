// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_non_deterministic_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_non_deterministic_type.carbon

var a: (i32, i32) = (2, 3);
var b: i32 = 0;
// CHECK:STDERR: fail_non_deterministic_type.carbon:[[@LINE+3]]:14: error: tuple index must be a constant [TupleIndexNotConstant]
// CHECK:STDERR: var c: i32 = a.(b);
// CHECK:STDERR:              ^~~~~
var c: i32 = a.(b);

// CHECK:STDOUT: --- fail_non_deterministic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.31, %.34) [template]
// CHECK:STDOUT:   %.35: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.35, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_9: init type = call constants.%Int(%.loc11_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_14: init type = call constants.%Int(%.loc11_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_9, %int.make_type_signed.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_signed.loc11_9, %.loc11_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_signed.loc11_14, %.loc11_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.3: type = converted %int.make_type_signed.loc12, %.loc12_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc16_8.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc16: init type = call constants.%Int(%.loc16_8.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_8.2: type = value_of_initializer %int.make_type_signed.loc16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_8.3: type = converted %int.make_type_signed.loc16, %.loc16_8.2 [template = constants.%i32]
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_22: Core.IntLiteral = int_value 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_25: Core.IntLiteral = int_value 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type.3 = tuple_literal (%.loc11_22, %.loc11_25)
// CHECK:STDOUT:   %.loc11_26.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_26.3: <bound method> = bound_method %.loc11_22, %.loc11_26.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_26.4: <specific function> = specific_function %.loc11_26.3, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.1: init %i32 = call %.loc11_26.4(%.loc11_22) [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_26.5: init %i32 = converted %.loc11_22, %int.convert_checked.loc11_26.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_26.6: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_26.7: init %i32 = initialize_from %.loc11_26.5 to %.loc11_26.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_26.8: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_26.9: <bound method> = bound_method %.loc11_25, %.loc11_26.8 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_26.10: <specific function> = specific_function %.loc11_26.9, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.2: init %i32 = call %.loc11_26.10(%.loc11_25) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_26.11: init %i32 = converted %.loc11_25, %int.convert_checked.loc11_26.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_26.12: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_26.13: init %i32 = initialize_from %.loc11_26.11 to %.loc11_26.12 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_26.14: init %tuple.type.2 = tuple_init (%.loc11_26.7, %.loc11_26.13) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_27: init %tuple.type.2 = converted %.loc11_26.1, %.loc11_26.14 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_27
// CHECK:STDOUT:   %.loc12_14: Core.IntLiteral = int_value 0 [template = constants.%.35]
// CHECK:STDOUT:   %.loc12_15.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_15.2: <bound method> = bound_method %.loc12_14, %.loc12_15.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_15.3: <specific function> = specific_function %.loc12_15.2, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_15.3(%.loc12_14) [template = constants.%.38]
// CHECK:STDOUT:   %.loc12_15.4: init %i32 = converted %.loc12_14, %int.convert_checked.loc12 [template = constants.%.38]
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15.4
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %b.ref: ref %i32 = name_ref b, file.%b
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
