// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_array_non_int_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_array_non_int_indexing.carbon

var a: [i32; 1] = (12,);
// CHECK:STDERR: fail_array_non_int_indexing.carbon:[[@LINE+6]]:16: error: cannot implicitly convert from `f64` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: var b: i32 = a[2.6];
// CHECK:STDERR:                ^~~
// CHECK:STDERR: fail_array_non_int_indexing.carbon:[[@LINE+3]]:16: note: type `f64` does not implement interface `ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: i32 = a[2.6];
// CHECK:STDERR:                ^~~
var b: i32 = a[2.6];

// CHECK:STDOUT: --- fail_array_non_int_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.1, %i32 [template]
// CHECK:STDOUT:   %int.12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int.12.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %array: %.1 = tuple_value (%int.12.2) [template]
// CHECK:STDOUT:   %.7: f64 = float_literal 2.6000000000000001 [template]
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
// CHECK:STDOUT:   %int.32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int.32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_signed.loc11, %.loc11_9.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %int.1, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %int.32.loc18: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc18: init type = call constants.%Int(%int.32.loc18) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_8.1: type = value_of_initializer %int.make_type_signed.loc18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_8.2: type = converted %int.make_type_signed.loc18, %.loc18_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.12: Core.IntLiteral = int_value 12 [template = constants.%int.12.1]
// CHECK:STDOUT:   %.loc11_23.1: %tuple.type = tuple_literal (%int.12)
// CHECK:STDOUT:   %.loc11_23.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int.12, %.loc11_23.2 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int.12) [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_23.3: init %i32 = converted %int.12, %int.convert_checked [template = constants.%int.12.2]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc11_23.4: ref %i32 = array_index file.%a.var, %int.0
// CHECK:STDOUT:   %.loc11_23.5: init %i32 = initialize_from %.loc11_23.3 to %.loc11_23.4 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_23.6: init %.1 = array_init (%.loc11_23.5) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_24: init %.1 = converted %.loc11_23.1, %.loc11_23.6 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_24
// CHECK:STDOUT:   %a.ref: ref %.1 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc18_16.1: f64 = float_literal 2.6000000000000001 [template = constants.%.7]
// CHECK:STDOUT:   %.loc18_16.2: %i32 = converted %.loc18_16.1, <error> [template = <error>]
// CHECK:STDOUT:   %.loc18_19.1: ref %i32 = array_index %a.ref, <error> [template = <error>]
// CHECK:STDOUT:   %.loc18_19.2: %i32 = bind_value %.loc18_19.1
// CHECK:STDOUT:   assign file.%b.var, %.loc18_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
