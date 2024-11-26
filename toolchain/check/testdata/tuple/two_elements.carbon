// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/two_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/two_elements.carbon

let v: (i32, i32) = (4, 102);
let w: (i32, i32) = v;

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 102 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 102 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.31, %.34) [template]
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
// CHECK:STDOUT:     .v = @__global_init.%v
// CHECK:STDOUT:     .w = @__global_init.%w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
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
// CHECK:STDOUT:   %.loc12_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_9: init type = call constants.%Int(%.loc12_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_14: init type = call constants.%Int(%.loc12_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc12_9, %int.make_type_signed.loc12_14)
// CHECK:STDOUT:   %.loc12_17.2: type = value_of_initializer %int.make_type_signed.loc12_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.3: type = converted %int.make_type_signed.loc12_9, %.loc12_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.4: type = value_of_initializer %int.make_type_signed.loc12_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.5: type = converted %int.make_type_signed.loc12_14, %.loc12_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_17.6: type = converted %.loc12_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc14_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_9: init type = call constants.%Int(%.loc14_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_14: init type = call constants.%Int(%.loc14_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_9, %int.make_type_signed.loc14_14)
// CHECK:STDOUT:   %.loc14_17.2: type = value_of_initializer %int.make_type_signed.loc14_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.3: type = converted %int.make_type_signed.loc14_9, %.loc14_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.4: type = value_of_initializer %int.make_type_signed.loc14_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.5: type = converted %int.make_type_signed.loc14_14, %.loc14_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_17.6: type = converted %.loc14_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT:   %.loc15_9: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15_9: init type = call constants.%Int(%.loc15_9) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_14: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15_14: init type = call constants.%Int(%.loc15_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc15_9, %int.make_type_signed.loc15_14)
// CHECK:STDOUT:   %.loc15_17.2: type = value_of_initializer %int.make_type_signed.loc15_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.3: type = converted %int.make_type_signed.loc15_9, %.loc15_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.4: type = value_of_initializer %int.make_type_signed.loc15_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.5: type = converted %int.make_type_signed.loc15_14, %.loc15_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_17.6: type = converted %.loc15_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %y.var: ref %tuple.type.2 = var y
// CHECK:STDOUT:   %y: ref %tuple.type.2 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_22: Core.IntLiteral = int_value 4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_25: Core.IntLiteral = int_value 102 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_28.1: %tuple.type.3 = tuple_literal (%.loc11_22, %.loc11_25)
// CHECK:STDOUT:   %.loc11_28.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_28.3: <bound method> = bound_method %.loc11_22, %.loc11_28.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_28.4: <specific function> = specific_function %.loc11_28.3, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.1: init %i32 = call %.loc11_28.4(%.loc11_22) [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_28.5: %i32 = value_of_initializer %int.convert_checked.loc11_28.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_28.6: %i32 = converted %.loc11_22, %.loc11_28.5 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_28.7: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_28.8: <bound method> = bound_method %.loc11_25, %.loc11_28.7 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_28.9: <specific function> = specific_function %.loc11_28.8, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.2: init %i32 = call %.loc11_28.9(%.loc11_25) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_28.10: %i32 = value_of_initializer %int.convert_checked.loc11_28.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_28.11: %i32 = converted %.loc11_25, %.loc11_28.10 [template = constants.%.34]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc11_28.6, %.loc11_28.11) [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_29: %tuple.type.2 = converted %.loc11_28.1, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   %v: %tuple.type.2 = bind_name v, %.loc11_29
// CHECK:STDOUT:   %v.ref: %tuple.type.2 = name_ref v, %v
// CHECK:STDOUT:   %w: %tuple.type.2 = bind_name w, %v.ref
// CHECK:STDOUT:   %.loc14_22: Core.IntLiteral = int_value 4 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_25: Core.IntLiteral = int_value 102 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14_28.1: %tuple.type.3 = tuple_literal (%.loc14_22, %.loc14_25)
// CHECK:STDOUT:   %.loc14_28.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_28.3: <bound method> = bound_method %.loc14_22, %.loc14_28.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc14_28.4: <specific function> = specific_function %.loc14_28.3, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.1: init %i32 = call %.loc14_28.4(%.loc14_22) [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_28.5: init %i32 = converted %.loc14_22, %int.convert_checked.loc14_28.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_28.6: ref %i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_28.7: init %i32 = initialize_from %.loc14_28.5 to %.loc14_28.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc14_28.8: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc14_28.9: <bound method> = bound_method %.loc14_25, %.loc14_28.8 [template = constants.%.32]
// CHECK:STDOUT:   %.loc14_28.10: <specific function> = specific_function %.loc14_28.9, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.2: init %i32 = call %.loc14_28.10(%.loc14_25) [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_28.11: init %i32 = converted %.loc14_25, %int.convert_checked.loc14_28.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_28.12: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_28.13: init %i32 = initialize_from %.loc14_28.11 to %.loc14_28.12 [template = constants.%.34]
// CHECK:STDOUT:   %.loc14_28.14: init %tuple.type.2 = tuple_init (%.loc14_28.7, %.loc14_28.13) to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_29: init %tuple.type.2 = converted %.loc14_28.1, %.loc14_28.14 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_29
// CHECK:STDOUT:   %x.ref: ref %tuple.type.2 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_21.1: ref %i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_21.2: %i32 = bind_value %.loc15_21.1
// CHECK:STDOUT:   %.loc15_21.3: ref %i32 = tuple_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_21.4: init %i32 = initialize_from %.loc15_21.2 to %.loc15_21.3
// CHECK:STDOUT:   %.loc15_21.5: ref %i32 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_21.6: %i32 = bind_value %.loc15_21.5
// CHECK:STDOUT:   %.loc15_21.7: ref %i32 = tuple_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_21.8: init %i32 = initialize_from %.loc15_21.6 to %.loc15_21.7
// CHECK:STDOUT:   %.loc15_21.9: init %tuple.type.2 = tuple_init (%.loc15_21.4, %.loc15_21.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_22: init %tuple.type.2 = converted %x.ref, %.loc15_21.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_22
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
