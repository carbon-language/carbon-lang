// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_access_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_access_error.carbon

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_access_error.carbon:[[@LINE+3]]:17: error: name `oops` not found [NameNotFound]
// CHECK:STDERR: var b: i32 = a.(oops);
// CHECK:STDERR:                 ^~~~
var b: i32 = a.(oops);

// CHECK:STDOUT: --- fail_access_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.471: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.8aa: <bound method> = bound_method %int_12.6a3, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.545: <specific function> = specific_function %Convert.bound.8aa, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.43d: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %Convert.bound.b1c: <bound method> = bound_method %int_6.462, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b26: <specific function> = specific_function %Convert.bound.b1c, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_6.b03: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.471 = tuple_value (%int_12.43d, %int_6.b03) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:   %a.var: ref %tuple.type.471 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.471 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.6a3]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [template = constants.%int_6.462]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type.f94 = tuple_literal (%int_12, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_27.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_27.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_27.1 [template = constants.%Convert.bound.8aa]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.1: <specific function> = specific_function %Convert.bound.loc11_27.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.545]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init %i32 = call %Convert.specific_fn.loc11_27.1(%int_12) [template = constants.%int_12.43d]
// CHECK:STDOUT:   %.loc11_27.2: init %i32 = converted %int_12, %int.convert_checked.loc11_27.1 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_27.3: init %i32 = initialize_from %.loc11_27.2 to %tuple.elem0 [template = constants.%int_12.43d]
// CHECK:STDOUT:   %impl.elem0.loc11_27.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc11_27.2: <bound method> = bound_method %int_6, %impl.elem0.loc11_27.2 [template = constants.%Convert.bound.b1c]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_27.2: <specific function> = specific_function %Convert.bound.loc11_27.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b26]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init %i32 = call %Convert.specific_fn.loc11_27.2(%int_6) [template = constants.%int_6.b03]
// CHECK:STDOUT:   %.loc11_27.4: init %i32 = converted %int_6, %int.convert_checked.loc11_27.2 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_27.5: init %i32 = initialize_from %.loc11_27.4 to %tuple.elem1 [template = constants.%int_6.b03]
// CHECK:STDOUT:   %.loc11_27.6: init %tuple.type.471 = tuple_init (%.loc11_27.3, %.loc11_27.5) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_28: init %tuple.type.471 = converted %.loc11_27.1, %.loc11_27.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_28
// CHECK:STDOUT:   %a.ref: ref %tuple.type.471 = name_ref a, file.%a
// CHECK:STDOUT:   %oops.ref: <error> = name_ref oops, <error> [template = <error>]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
