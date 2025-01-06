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
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+3]]:17: note: type `f64` does not implement interface `Core.ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
var b: i32 = a.(2.6);

// CHECK:STDOUT: --- fail_non_int_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_12.6: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_6.4: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.f: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.c3: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.bd: <witness> = impl_witness (imports.%import_ref.05), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.2c: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.54: %Convert.type.2c = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.0a: <bound method> = bound_method %int_12.6, %Convert.54 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.0a, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.5: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %Convert.bound.03: <bound method> = bound_method %int_6.4, %Convert.54 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.03, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.0: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.5 = tuple_value (%int_12.5, %int_6.0) [template]
// CHECK:STDOUT:   %float: f64 = float_literal 2.6000000000000001 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.30a
// CHECK:STDOUT:     .ImplicitAs = %import_ref.41f
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
// CHECK:STDOUT:   %a.var: ref %tuple.type.5 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.5 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.6]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.4]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type.f = tuple_literal (%int_12, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_27.1: %Convert.type.c3 = impl_witness_access constants.%impl_witness.bd, element0 [template = constants.%Convert.54]
// CHECK:STDOUT:   %Convert.bound.loc11_27.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_27.1 [template = constants.%Convert.bound.0a]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.1: <specific function> = specific_function %Convert.bound.loc11_27.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init %i32 = call %Convert.specific_fn.loc11_27.1(%int_12) [template = constants.%int_12.5]
// CHECK:STDOUT:   %.loc11_27.2: init %i32 = converted %int_12, %int.convert_checked.loc11_27.1 [template = constants.%int_12.5]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_27.3: init %i32 = initialize_from %.loc11_27.2 to %tuple.elem0 [template = constants.%int_12.5]
// CHECK:STDOUT:   %impl.elem0.loc11_27.2: %Convert.type.c3 = impl_witness_access constants.%impl_witness.bd, element0 [template = constants.%Convert.54]
// CHECK:STDOUT:   %Convert.bound.loc11_27.2: <bound method> = bound_method %int_6, %impl.elem0.loc11_27.2 [template = constants.%Convert.bound.03]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.2: <specific function> = specific_function %Convert.bound.loc11_27.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init %i32 = call %Convert.specific_fn.loc11_27.2(%int_6) [template = constants.%int_6.0]
// CHECK:STDOUT:   %.loc11_27.4: init %i32 = converted %int_6, %int.convert_checked.loc11_27.2 [template = constants.%int_6.0]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_27.5: init %i32 = initialize_from %.loc11_27.4 to %tuple.elem1 [template = constants.%int_6.0]
// CHECK:STDOUT:   %.loc11_27.6: init %tuple.type.5 = tuple_init (%.loc11_27.3, %.loc11_27.5) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_28: init %tuple.type.5 = converted %.loc11_27.1, %.loc11_27.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %tuple.type.5 = name_ref a, file.%a
// CHECK:STDOUT:   %float: f64 = float_literal 2.6000000000000001 [template = constants.%float]
// CHECK:STDOUT:   %.loc18: Core.IntLiteral = converted %float, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
