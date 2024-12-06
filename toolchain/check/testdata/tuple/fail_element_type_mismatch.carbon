// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon

// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+6]]:21: error: cannot implicitly convert from `f64` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+3]]:21: note: type `f64` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                     ^~~~~~~~~~
var x: (i32, i32) = (2, 65.89);

// CHECK:STDOUT: --- fail_element_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %float: f64 = float_literal 65.890000000000001 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, f64) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc17_9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17_9: init type = call constants.%Int(%int_32.loc17_9) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc17_14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17_14: init type = call constants.%Int(%int_32.loc17_14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc17_9, %int.make_type_signed.loc17_14)
// CHECK:STDOUT:   %.loc17_17.2: type = value_of_initializer %int.make_type_signed.loc17_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.3: type = converted %int.make_type_signed.loc17_9, %.loc17_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.4: type = value_of_initializer %int.make_type_signed.loc17_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.5: type = converted %int.make_type_signed.loc17_14, %.loc17_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_17.6: type = converted %.loc17_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %x.var: ref %tuple.type.2 = var x
// CHECK:STDOUT:   %x: ref %tuple.type.2 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %float: f64 = float_literal 65.890000000000001 [template = constants.%float]
// CHECK:STDOUT:   %.loc17_30.1: %tuple.type.3 = tuple_literal (%int_2, %float)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc17_30.2: init %i32 = converted %int_2, %int.convert_checked [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc17_30.3: init %i32 = initialize_from %.loc17_30.2 to %tuple.elem0 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc17_30.4: %i32 = converted %float, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
