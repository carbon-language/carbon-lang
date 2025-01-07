// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_negative_indexing.carbon

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:14: error: tuple element index `-10` is past the end of type `(i32, i32)` [TupleIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a.(-10);
// CHECK:STDERR:              ^~~~~~~
var b: i32 = a.(-10);

// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_6.1: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.19: <witness> = impl_witness (imports.%import_ref.14), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_12.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_6.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.2: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_12.2, %int_6.2) [template]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Op.type.13: type = fn_type @Op.13 [template]
// CHECK:STDOUT:   %impl_witness.21: <witness> = impl_witness (imports.%import_ref.254) [template]
// CHECK:STDOUT:   %Op.type.14: type = fn_type @Op.14 [template]
// CHECK:STDOUT:   %Op.14: %Op.type.14 = struct_value () [template]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_10, %Op.14 [template]
// CHECK:STDOUT:   %int_-10: Core.IntLiteral = int_value -10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
// CHECK:STDOUT:     .Negate = %import_ref.252
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
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.1]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.1]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type.3 = tuple_literal (%int_12, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_27.1: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_27.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_27.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.1: <specific function> = specific_function %Convert.bound.loc11_27.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init %i32 = call %Convert.specific_fn.loc11_27.1(%int_12) [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc11_27.2: init %i32 = converted %int_12, %int.convert_checked.loc11_27.1 [template = constants.%int_12.2]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_27.3: init %i32 = initialize_from %.loc11_27.2 to %tuple.elem0 [template = constants.%int_12.2]
// CHECK:STDOUT:   %impl.elem0.loc11_27.2: %Convert.type.2 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc11_27.2: <bound method> = bound_method %int_6, %impl.elem0.loc11_27.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.2: <specific function> = specific_function %Convert.bound.loc11_27.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init %i32 = call %Convert.specific_fn.loc11_27.2(%int_6) [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_27.4: init %i32 = converted %int_6, %int.convert_checked.loc11_27.2 [template = constants.%int_6.2]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_27.5: init %i32 = initialize_from %.loc11_27.4 to %tuple.elem1 [template = constants.%int_6.2]
// CHECK:STDOUT:   %.loc11_27.6: init %tuple.type.2 = tuple_init (%.loc11_27.3, %.loc11_27.5) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_28: init %tuple.type.2 = converted %.loc11_27.1, %.loc11_27.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10]
// CHECK:STDOUT:   %impl.elem0.loc15: %Op.type.13 = impl_witness_access constants.%impl_witness.21, element0 [template = constants.%Op.14]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_10, %impl.elem0.loc15 [template = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %Op.bound(%int_10) [template = constants.%int_-10]
// CHECK:STDOUT:   %.loc15_17.1: Core.IntLiteral = value_of_initializer %int.snegate [template = constants.%int_-10]
// CHECK:STDOUT:   %.loc15_17.2: Core.IntLiteral = converted %int.snegate, %.loc15_17.1 [template = constants.%int_-10]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
