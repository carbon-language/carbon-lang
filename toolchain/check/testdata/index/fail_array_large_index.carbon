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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.1.1, %i32 [template]
// CHECK:STDOUT:   %int.12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.12.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %.18, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %array: %.1 = tuple_value (%int.12.2) [template]
// CHECK:STDOUT:   %.19: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %.19, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int.2147483647.1: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %.20: <bound method> = bound_method %int.2147483647.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %.20, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2147483647.2: %i32 = int_value 2147483647 [template]
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
// CHECK:STDOUT:   %int.32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int.32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed.loc11, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %int.1, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%int.32.loc17) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_8.2: type = converted %int.make_type_signed.loc17, %.loc17_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %int.32.loc22: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc22: init type = call constants.%Int(%int.32.loc22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_8.1: type = value_of_initializer %int.make_type_signed.loc22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc22_8.2: type = converted %int.make_type_signed.loc22, %.loc22_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.12: Core.IntLiteral = int_value 12 [template = constants.%int.12.1]
// CHECK:STDOUT:   %.loc11_23.1: %tuple.type = tuple_literal (%int.12)
// CHECK:STDOUT:   %.loc11_23.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_23.3: <bound method> = bound_method %int.12, %.loc11_23.2 [template = constants.%.18]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %.loc11_23.3, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int.12) [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_23.4: init %i32 = converted %int.12, %int.convert_checked.loc11 [template = constants.%int.12.2]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc11_23.5: ref %i32 = array_index file.%a.var, %int.0
// CHECK:STDOUT:   %.loc11_23.6: init %i32 = initialize_from %.loc11_23.4 to %.loc11_23.5 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_23.7: init %.1 = array_init (%.loc11_23.6) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_24: init %.1 = converted %.loc11_23.1, %.loc11_23.7 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_24
// CHECK:STDOUT:   %a.ref.loc17: ref %.1 = name_ref a, file.%a
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc17_16.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_16.2: <bound method> = bound_method %int.1, %.loc17_16.1 [template = constants.%.19]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %.loc17_16.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %Convert.specific_fn.loc17(%int.1) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc17_16.3: %i32 = value_of_initializer %int.convert_checked.loc17 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc17_16.4: %i32 = converted %int.1, %.loc17_16.3 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc17_17.1: ref %i32 = array_index %a.ref.loc17, %.loc17_16.4 [template = <error>]
// CHECK:STDOUT:   %.loc17_17.2: %i32 = bind_value %.loc17_17.1
// CHECK:STDOUT:   assign file.%b.var, %.loc17_17.2
// CHECK:STDOUT:   %a.ref.loc22: ref %.1 = name_ref a, file.%a
// CHECK:STDOUT:   %int.2147483647: Core.IntLiteral = int_value 2147483647 [template = constants.%int.2147483647.1]
// CHECK:STDOUT:   %.loc22_16.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc22_16.2: <bound method> = bound_method %int.2147483647, %.loc22_16.1 [template = constants.%.20]
// CHECK:STDOUT:   %Convert.specific_fn.loc22: <specific function> = specific_function %.loc22_16.2, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %Convert.specific_fn.loc22(%int.2147483647) [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc22_16.3: %i32 = value_of_initializer %int.convert_checked.loc22 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc22_16.4: %i32 = converted %int.2147483647, %.loc22_16.3 [template = constants.%int.2147483647.2]
// CHECK:STDOUT:   %.loc22_27.1: ref %i32 = array_index %a.ref.loc22, %.loc22_16.4 [template = <error>]
// CHECK:STDOUT:   %.loc22_27.2: %i32 = bind_value %.loc22_27.1
// CHECK:STDOUT:   assign file.%c.var, %.loc22_27.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
