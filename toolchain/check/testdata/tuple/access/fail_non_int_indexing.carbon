// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_non_int_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_non_int_indexing.carbon

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+6]]:17: error: cannot implicitly convert from `f64` to `Core.IntLiteral` [ImplicitAsConversionFailure]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+3]]:17: note: type `f64` does not implement interface `ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
var b: i32 = a.(2.6);

// CHECK:STDOUT: --- fail_non_int_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int.12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int.6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.17: <bound method> = bound_method %int.12.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %.17, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %.18: <bound method> = bound_method %int.6.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %.18, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int.12.2, %int.6.2) [template]
// CHECK:STDOUT:   %.19: f64 = float_literal 2.6000000000000001 [template]
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
// CHECK:STDOUT:   %.loc11_17.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_9, %int.make_type_signed.loc11_14)
// CHECK:STDOUT:   %.loc11_17.2: type = value_of_initializer %int.make_type_signed.loc11_9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.3: type = converted %int.make_type_signed.loc11_9, %.loc11_17.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.4: type = value_of_initializer %int.make_type_signed.loc11_14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.5: type = converted %int.make_type_signed.loc11_14, %.loc11_17.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_17.6: type = converted %.loc11_17.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
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
// CHECK:STDOUT:   %int.6: Core.IntLiteral = int_value 6 [template = constants.%int.6.1]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type.3 = tuple_literal (%int.12, %int.6)
// CHECK:STDOUT:   %.loc11_27.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_27.3: <bound method> = bound_method %int.12, %.loc11_27.2 [template = constants.%.17]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.1: <specific function> = specific_function %.loc11_27.3, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init %i32 = call %Convert.specific_fn.loc11_27.1(%int.12) [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_27.4: init %i32 = converted %int.12, %int.convert_checked.loc11_27.1 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_27.5: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_27.6: init %i32 = initialize_from %.loc11_27.4 to %.loc11_27.5 [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc11_27.7: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_27.8: <bound method> = bound_method %int.6, %.loc11_27.7 [template = constants.%.18]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.2: <specific function> = specific_function %.loc11_27.8, @Convert.2(constants.%int.32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init %i32 = call %Convert.specific_fn.loc11_27.2(%int.6) [template = constants.%int.6.2]
// CHECK:STDOUT:   %.loc11_27.9: init %i32 = converted %int.6, %int.convert_checked.loc11_27.2 [template = constants.%int.6.2]
// CHECK:STDOUT:   %.loc11_27.10: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_27.11: init %i32 = initialize_from %.loc11_27.9 to %.loc11_27.10 [template = constants.%int.6.2]
// CHECK:STDOUT:   %.loc11_27.12: init %tuple.type.2 = tuple_init (%.loc11_27.6, %.loc11_27.11) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_28: init %tuple.type.2 = converted %.loc11_27.1, %.loc11_27.12 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc18_17.1: f64 = float_literal 2.6000000000000001 [template = constants.%.19]
// CHECK:STDOUT:   %.loc18_17.2: Core.IntLiteral = converted %.loc18_17.1, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
