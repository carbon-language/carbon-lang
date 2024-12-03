// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/assign_var.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/assign_var.carbon

var a: (i32, i32, i32) = (1, 2, 3);
var b: [i32; 3] = a;

// CHECK:STDOUT: --- assign_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int.3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int.1.2, %int.2.2, %int.3.2) [template]
// CHECK:STDOUT:   %.17: type = array_type %int.3.1, %i32 [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32.loc11_9: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_9: init type = call constants.%Int(%int.32.loc11_9) [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc11_14: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_14: init type = call constants.%Int(%int.32.loc11_14) [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc11_19: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_19: init type = call constants.%Int(%int.32.loc11_19) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_9, %int.make_type_signed.loc11_14, %int.make_type_signed.loc11_19)
// CHECK:STDOUT:   %.loc11_22.2: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.3: type = converted %int.make_type_signed.loc11_9, %.loc11_22.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.4: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.5: type = converted %int.make_type_signed.loc11_14, %.loc11_22.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.6: type = value_of_initializer %int.make_type_signed.loc11_19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.7: type = converted %int.make_type_signed.loc11_19, %.loc11_22.6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.8: type = converted %.loc11_22.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int.32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int.32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc12_9.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_9.2: type = converted %int.make_type_signed.loc12, %.loc12_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15: type = array_type %int.3, %i32 [template = constants.%.17]
// CHECK:STDOUT:   %b.var: ref %.17 = var b
// CHECK:STDOUT:   %b: ref %.17 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2.loc11: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc11_34.1: %tuple.type.3 = tuple_literal (%int.1.loc11, %int.2.loc11, %int.3)
// CHECK:STDOUT:   %.loc11_34.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_34.1: <bound method> = bound_method %int.1.loc11, %.loc11_34.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_34.1: <specific function> = specific_function %Convert.bound.loc11_34.1, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.1: init %i32 = call %Convert.specific_fn.loc11_34.1(%int.1.loc11) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_34.3: init %i32 = converted %int.1.loc11, %int.convert_checked.loc11_34.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_34.4: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_34.5: init %i32 = initialize_from %.loc11_34.3 to %.loc11_34.4 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc11_34.6: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_34.2: <bound method> = bound_method %int.2.loc11, %.loc11_34.6 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_34.2: <specific function> = specific_function %Convert.bound.loc11_34.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.2: init %i32 = call %Convert.specific_fn.loc11_34.2(%int.2.loc11) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_34.7: init %i32 = converted %int.2.loc11, %int.convert_checked.loc11_34.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_34.8: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_34.9: init %i32 = initialize_from %.loc11_34.7 to %.loc11_34.8 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc11_34.10: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_34.3: <bound method> = bound_method %int.3, %.loc11_34.10 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_34.3: <specific function> = specific_function %Convert.bound.loc11_34.3, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.3: init %i32 = call %Convert.specific_fn.loc11_34.3(%int.3) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_34.11: init %i32 = converted %int.3, %int.convert_checked.loc11_34.3 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_34.12: ref %i32 = tuple_access file.%a.var, element2
// CHECK:STDOUT:   %.loc11_34.13: init %i32 = initialize_from %.loc11_34.11 to %.loc11_34.12 [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc11_34.14: init %tuple.type.2 = tuple_init (%.loc11_34.5, %.loc11_34.9, %.loc11_34.13) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_35: init %tuple.type.2 = converted %.loc11_34.1, %.loc11_34.14 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_35
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_19.1: ref %i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_19.2: %i32 = bind_value %.loc12_19.1
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc12_19.3: ref %i32 = array_index file.%b.var, %int.0
// CHECK:STDOUT:   %.loc12_19.4: init %i32 = initialize_from %.loc12_19.2 to %.loc12_19.3
// CHECK:STDOUT:   %.loc12_19.5: ref %i32 = tuple_access %a.ref, element1
// CHECK:STDOUT:   %.loc12_19.6: %i32 = bind_value %.loc12_19.5
// CHECK:STDOUT:   %int.1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc12_19.7: ref %i32 = array_index file.%b.var, %int.1.loc12
// CHECK:STDOUT:   %.loc12_19.8: init %i32 = initialize_from %.loc12_19.6 to %.loc12_19.7
// CHECK:STDOUT:   %.loc12_19.9: ref %i32 = tuple_access %a.ref, element2
// CHECK:STDOUT:   %.loc12_19.10: %i32 = bind_value %.loc12_19.9
// CHECK:STDOUT:   %int.2.loc12: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc12_19.11: ref %i32 = array_index file.%b.var, %int.2.loc12
// CHECK:STDOUT:   %.loc12_19.12: init %i32 = initialize_from %.loc12_19.10 to %.loc12_19.11
// CHECK:STDOUT:   %.loc12_19.13: init %.17 = array_init (%.loc12_19.4, %.loc12_19.8, %.loc12_19.12) to file.%b.var
// CHECK:STDOUT:   %.loc12_20: init %.17 = converted %a.ref, %.loc12_19.13
// CHECK:STDOUT:   assign file.%b.var, %.loc12_20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
