// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon

var c: [i32; 2] = (42, 42);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:16: error: cannot access member of interface `Negate` in type `Core.IntLiteral` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: var d: i32 = c[-10];
// CHECK:STDERR:                ^~~
var d: i32 = c[-10];


// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.2, %i32 [template]
// CHECK:STDOUT:   %int.42.1: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.42.1, %Convert.14 [template]
// CHECK:STDOUT:   %.19: <specific function> = specific_function %.18, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.42.2: %i32 = int_value 42 [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array: %.1 = tuple_value (%int.42.2, %int.42.2) [template]
// CHECK:STDOUT:   %int.10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Negate = %import_ref.38
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int.32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int.2: Core.IntLiteral = int_value 2 [template = constants.%int.2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed.loc11, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %int.2, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %c.var: ref %.1 = var c
// CHECK:STDOUT:   %c: ref %.1 = bind_name c, %c.var
// CHECK:STDOUT:   %int.32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int.32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_signed.loc15, %.loc15_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.42.loc11_20: Core.IntLiteral = int_value 42 [template = constants.%int.42.1]
// CHECK:STDOUT:   %int.42.loc11_24: Core.IntLiteral = int_value 42 [template = constants.%int.42.1]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type = tuple_literal (%int.42.loc11_20, %int.42.loc11_24)
// CHECK:STDOUT:   %.loc11_26.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_26.3: <bound method> = bound_method %int.42.loc11_20, %.loc11_26.2 [template = constants.%.18]
// CHECK:STDOUT:   %.loc11_26.4: <specific function> = specific_function %.loc11_26.3, @Convert.2(constants.%int.32) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.1: init %i32 = call %.loc11_26.4(%int.42.loc11_20) [template = constants.%int.42.2]
// CHECK:STDOUT:   %.loc11_26.5: init %i32 = converted %int.42.loc11_20, %int.convert_checked.loc11_26.1 [template = constants.%int.42.2]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc11_26.6: ref %i32 = array_index file.%c.var, %int.0
// CHECK:STDOUT:   %.loc11_26.7: init %i32 = initialize_from %.loc11_26.5 to %.loc11_26.6 [template = constants.%int.42.2]
// CHECK:STDOUT:   %.loc11_26.8: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_26.9: <bound method> = bound_method %int.42.loc11_24, %.loc11_26.8 [template = constants.%.18]
// CHECK:STDOUT:   %.loc11_26.10: <specific function> = specific_function %.loc11_26.9, @Convert.2(constants.%int.32) [template = constants.%.19]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.2: init %i32 = call %.loc11_26.10(%int.42.loc11_24) [template = constants.%int.42.2]
// CHECK:STDOUT:   %.loc11_26.11: init %i32 = converted %int.42.loc11_24, %int.convert_checked.loc11_26.2 [template = constants.%int.42.2]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc11_26.12: ref %i32 = array_index file.%c.var, %int.1
// CHECK:STDOUT:   %.loc11_26.13: init %i32 = initialize_from %.loc11_26.11 to %.loc11_26.12 [template = constants.%int.42.2]
// CHECK:STDOUT:   %.loc11_26.14: init %.1 = array_init (%.loc11_26.7, %.loc11_26.13) to file.%c.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_27: init %.1 = converted %.loc11_26.1, %.loc11_26.14 [template = constants.%array]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_27
// CHECK:STDOUT:   %c.ref: ref %.1 = name_ref c, file.%c
// CHECK:STDOUT:   %int.10: Core.IntLiteral = int_value 10 [template = constants.%int.10]
// CHECK:STDOUT:   %.loc15_19.1: ref %i32 = array_index %c.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc15_19.2: %i32 = bind_value %.loc15_19.1
// CHECK:STDOUT:   assign file.%d.var, %.loc15_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
