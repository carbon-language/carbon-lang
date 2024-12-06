// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/two_entries.carbon

let v: {.a: i32, .b: i32} = {.a = 1, .b = 2};
let w: {.a: i32, .b: i32} = v;

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: --- two_entries.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%int_1.2, %int_2.2) [template]
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
// CHECK:STDOUT:   %int_32.loc11_13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_13: init type = call constants.%Int(%int_32.loc11_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_signed.loc11_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_signed.loc11_13, %.loc11_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_22: init type = call constants.%Int(%int_32.loc11_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.1: type = value_of_initializer %int.make_type_signed.loc11_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_22.2: type = converted %int.make_type_signed.loc11_22, %.loc11_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b.loc11: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %int_32.loc12_13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_13: init type = call constants.%Int(%int_32.loc12_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.1: type = value_of_initializer %int.make_type_signed.loc12_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_13.2: type = converted %int.make_type_signed.loc12_13, %.loc12_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc12_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_22: init type = call constants.%Int(%int_32.loc12_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_22.1: type = value_of_initializer %int.make_type_signed.loc12_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_22.2: type = converted %int.make_type_signed.loc12_22, %.loc12_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b.loc12: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %int_32.loc14_13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_13: init type = call constants.%Int(%int_32.loc14_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_13.1: type = value_of_initializer %int.make_type_signed.loc14_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_13.2: type = converted %int.make_type_signed.loc14_13, %.loc14_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_22: init type = call constants.%Int(%int_32.loc14_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_22.1: type = value_of_initializer %int.make_type_signed.loc14_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_22.2: type = converted %int.make_type_signed.loc14_22, %.loc14_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b.loc14: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.1 = var x
// CHECK:STDOUT:   %x: ref %struct_type.a.b.1 = bind_name x, %x.var
// CHECK:STDOUT:   %int_32.loc15_13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_13: init type = call constants.%Int(%int_32.loc15_13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.1: type = value_of_initializer %int.make_type_signed.loc15_13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.2: type = converted %int.make_type_signed.loc15_13, %.loc15_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc15_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_22: init type = call constants.%Int(%int_32.loc15_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_22.1: type = value_of_initializer %int.make_type_signed.loc15_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_22.2: type = converted %int.make_type_signed.loc15_22, %.loc15_22.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b.loc15: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.1 = var y
// CHECK:STDOUT:   %y: ref %struct_type.a.b.1 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc11_44.1: %struct_type.a.b.2 = struct_literal (%int_1.loc11, %int_2.loc11)
// CHECK:STDOUT:   %impl.elem0.loc11_44.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_44.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_44.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_44.1: <specific function> = specific_function %Convert.bound.loc11_44.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.1: init %i32 = call %Convert.specific_fn.loc11_44.1(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_44.2: %i32 = value_of_initializer %int.convert_checked.loc11_44.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_44.3: %i32 = converted %int_1.loc11, %.loc11_44.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_44.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_44.2: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_44.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_44.2: <specific function> = specific_function %Convert.bound.loc11_44.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.2: init %i32 = call %Convert.specific_fn.loc11_44.2(%int_2.loc11) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_44.4: %i32 = value_of_initializer %int.convert_checked.loc11_44.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_44.5: %i32 = converted %int_2.loc11, %.loc11_44.4 [template = constants.%int_2.2]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%.loc11_44.3, %.loc11_44.5) [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_45: %struct_type.a.b.1 = converted %.loc11_44.1, %struct [template = constants.%struct]
// CHECK:STDOUT:   %v: %struct_type.a.b.1 = bind_name v, %.loc11_45
// CHECK:STDOUT:   %v.ref: %struct_type.a.b.1 = name_ref v, %v
// CHECK:STDOUT:   %w: %struct_type.a.b.1 = bind_name w, %v.ref
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc14_44.1: %struct_type.a.b.2 = struct_literal (%int_1.loc14, %int_2.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_44.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_44.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_44.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_44.1: <specific function> = specific_function %Convert.bound.loc14_44.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.1: init %i32 = call %Convert.specific_fn.loc14_44.1(%int_1.loc14) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_44.2: init %i32 = converted %int_1.loc14, %int.convert_checked.loc14_44.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_44.3: ref %i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_44.4: init %i32 = initialize_from %.loc14_44.2 to %.loc14_44.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc14_44.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc14_44.2: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_44.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_44.2: <specific function> = specific_function %Convert.bound.loc14_44.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.2: init %i32 = call %Convert.specific_fn.loc14_44.2(%int_2.loc14) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_44.5: init %i32 = converted %int_2.loc14, %int.convert_checked.loc14_44.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_44.6: ref %i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_44.7: init %i32 = initialize_from %.loc14_44.5 to %.loc14_44.6 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc14_44.8: init %struct_type.a.b.1 = struct_init (%.loc14_44.4, %.loc14_44.7) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc14_45: init %struct_type.a.b.1 = converted %.loc14_44.1, %.loc14_44.8 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_45
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.1 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_29.1: ref %i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_29.2: %i32 = bind_value %.loc15_29.1
// CHECK:STDOUT:   %.loc15_29.3: ref %i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_29.4: init %i32 = initialize_from %.loc15_29.2 to %.loc15_29.3
// CHECK:STDOUT:   %.loc15_29.5: ref %i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_29.6: %i32 = bind_value %.loc15_29.5
// CHECK:STDOUT:   %.loc15_29.7: ref %i32 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_29.8: init %i32 = initialize_from %.loc15_29.6 to %.loc15_29.7
// CHECK:STDOUT:   %.loc15_29.9: init %struct_type.a.b.1 = struct_init (%.loc15_29.4, %.loc15_29.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_30: init %struct_type.a.b.1 = converted %x.ref, %.loc15_29.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_30
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
