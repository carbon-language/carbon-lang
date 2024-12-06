// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_array_large_index.carbon

var a: [i32; 1] = (12,);

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+4]]:16: error: array index `1` is past the end of type `[i32; 1]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a[1];
// CHECK:STDERR:                ^
// CHECK:STDERR:
var b: i32 = a[1];

// CHECK:STDERR: fail_array_large_index.carbon:[[@LINE+3]]:16: error: array index `2147483647` is past the end of type `[i32; 1]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var c: i32 = a[0x7FFF_FFFF];
// CHECK:STDERR:                ^~~~~~~~~~~
var c: i32 = a[0x7FFF_FFFF];

// CHECK:STDOUT: --- fail_array_large_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.1, %i32 [template]
// CHECK:STDOUT:   %int_12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_12.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_12.2) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_2147483647.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2147483647.2: %i32 = int_value 2147483647 [template]
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
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed.loc11, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%int_32.loc17) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.2: type = converted %int.make_type_signed.loc17, %.loc17_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%int_32.loc22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_8.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_8.2: type = converted %int.make_type_signed.loc22, %.loc22_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.1]
// CHECK:STDOUT:   %.loc11_23.1: %tuple.type = tuple_literal (%int_12)
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_12, %impl.elem0.loc11 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_12) [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc11_23.2: init %i32 = converted %int_12, %int.convert_checked.loc11 [template = constants.%int_12.2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_23.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_23.4: init %i32 = initialize_from %.loc11_23.2 to %.loc11_23.3 [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc11_23.5: init %array_type = array_init (%.loc11_23.4) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_24: init %array_type = converted %.loc11_23.1, %.loc11_23.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_24
// CHECK:STDOUT:   %a.ref.loc17: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%int_32.loc17) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.2: type = converted %int.make_type_signed.loc17, %.loc17_17.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc17: <bound method> = bound_method %int_1, %impl.elem0.loc17 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %Convert.bound.loc17, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %Convert.specific_fn.loc17(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc17_16.1: %i32 = value_of_initializer %int.convert_checked.loc17 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc17_16.2: %i32 = converted %int_1, %.loc17_16.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc17_17.3: ref %i32 = array_index %a.ref.loc17, %.loc17_16.2 [template = <error>]
// CHECK:STDOUT:   %.loc17_17.4: %i32 = bind_value %.loc17_17.3
// CHECK:STDOUT:   assign file.%b.var, %.loc17_17.4
// CHECK:STDOUT:   %a.ref.loc22: ref %array_type = name_ref a, file.%a
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int_2147483647.1]
// CHECK:STDOUT:   %int_32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%int_32.loc22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_27.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_27.2: type = converted %int.make_type_signed.loc22, %.loc22_27.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc22: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc22: <bound method> = bound_method %int_2147483647, %impl.elem0.loc22 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc22: <specific function> = specific_function %Convert.bound.loc22, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %Convert.specific_fn.loc22(%int_2147483647) [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc22_16.1: %i32 = value_of_initializer %int.convert_checked.loc22 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc22_16.2: %i32 = converted %int_2147483647, %.loc22_16.1 [template = constants.%int_2147483647.2]
// CHECK:STDOUT:   %.loc22_27.3: ref %i32 = array_index %a.ref.loc22, %.loc22_16.2 [template = <error>]
// CHECK:STDOUT:   %.loc22_27.4: %i32 = bind_value %.loc22_27.3
// CHECK:STDOUT:   assign file.%c.var, %.loc22_27.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
