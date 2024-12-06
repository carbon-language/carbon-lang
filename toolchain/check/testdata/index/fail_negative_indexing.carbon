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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [template]
// CHECK:STDOUT:   %int_42.1: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_42.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_42.2: %i32 = int_value 42 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_42.2, %int_42.2) [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
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
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed.loc11, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %c.var: ref %array_type = var c
// CHECK:STDOUT:   %c: ref %array_type = bind_name c, %c.var
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_signed.loc15, %.loc15_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_42.loc11_20: Core.IntLiteral = int_value 42 [template = constants.%int_42.1]
// CHECK:STDOUT:   %int_42.loc11_24: Core.IntLiteral = int_value 42 [template = constants.%int_42.1]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type = tuple_literal (%int_42.loc11_20, %int_42.loc11_24)
// CHECK:STDOUT:   %impl.elem0.loc11_26.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_26.1: <bound method> = bound_method %int_42.loc11_20, %impl.elem0.loc11_26.1 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_26.1: <specific function> = specific_function %Convert.bound.loc11_26.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.1: init %i32 = call %Convert.specific_fn.loc11_26.1(%int_42.loc11_20) [template = constants.%int_42.2]
// CHECK:STDOUT:   %.loc11_26.2: init %i32 = converted %int_42.loc11_20, %int.convert_checked.loc11_26.1 [template = constants.%int_42.2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc11_26.3: ref %i32 = array_index file.%c.var, %int_0
// CHECK:STDOUT:   %.loc11_26.4: init %i32 = initialize_from %.loc11_26.2 to %.loc11_26.3 [template = constants.%int_42.2]
// CHECK:STDOUT:   %impl.elem0.loc11_26.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_26.2: <bound method> = bound_method %int_42.loc11_24, %impl.elem0.loc11_26.2 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_26.2: <specific function> = specific_function %Convert.bound.loc11_26.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked.loc11_26.2: init %i32 = call %Convert.specific_fn.loc11_26.2(%int_42.loc11_24) [template = constants.%int_42.2]
// CHECK:STDOUT:   %.loc11_26.5: init %i32 = converted %int_42.loc11_24, %int.convert_checked.loc11_26.2 [template = constants.%int_42.2]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc11_26.6: ref %i32 = array_index file.%c.var, %int_1
// CHECK:STDOUT:   %.loc11_26.7: init %i32 = initialize_from %.loc11_26.5 to %.loc11_26.6 [template = constants.%int_42.2]
// CHECK:STDOUT:   %.loc11_26.8: init %array_type = array_init (%.loc11_26.4, %.loc11_26.7) to file.%c.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_27: init %array_type = converted %.loc11_26.1, %.loc11_26.8 [template = constants.%array]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_27
// CHECK:STDOUT:   %c.ref: ref %array_type = name_ref c, file.%c
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.2: type = converted %int.make_type_signed, %.loc15_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.3: ref %i32 = array_index %c.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc15_19.4: %i32 = bind_value %.loc15_19.3
// CHECK:STDOUT:   assign file.%d.var, %.loc15_19.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
