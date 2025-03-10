// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_element_type_mismatch.carbon

// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+7]]:21: error: cannot implicitly convert value of type `f64` to `i32` [ImplicitAsConversionFailure]
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+4]]:21: note: type `f64` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                     ^~~~~~~~~~
// CHECK:STDERR:
var x: (i32, i32) = (2, 65.89);

// CHECK:STDOUT: --- fail_element_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 65.890000000000001 [concrete]
// CHECK:STDOUT:   %tuple.type.2c1: type = tuple_type (Core.IntLiteral, f64) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %tuple.type.d07 = binding_pattern x
// CHECK:STDOUT:     %.loc18_1: %tuple.type.d07 = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type.d07 = var x
// CHECK:STDOUT:   %.loc18_17.1: type = splice_block %.loc18_17.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc18_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_17.2: %tuple.type.24b = tuple_literal (%i32.loc18_9, %i32.loc18_14)
// CHECK:STDOUT:     %.loc18_17.3: type = converted %.loc18_17.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.d07 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %float: f64 = float_literal 65.890000000000001 [concrete = constants.%float]
// CHECK:STDOUT:   %.loc18_30.1: %tuple.type.2c1 = tuple_literal (%int_2, %float)
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_30.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_30.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc18_30.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_30.2: init %i32 = converted %int_2, %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc18_30.3: init %i32 = initialize_from %.loc18_30.2 to %tuple.elem0 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_30.4: %i32 = converted %float, <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
