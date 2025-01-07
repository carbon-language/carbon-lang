// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/fail_non_member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/fail_non_member_access.carbon

var x: {.a: i32} = {.a = 4};
// CHECK:STDERR: fail_non_member_access.carbon:[[@LINE+3]]:14: error: type `{.a: i32}` does not have a member `b` [QualifiedExprNameNotFound]
// CHECK:STDERR: var y: i32 = x.b;
// CHECK:STDERR:              ^~~
var y: i32 = x.b;

// CHECK:STDOUT: --- fail_non_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.a.7cc1: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %int_4.0c1b: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %struct_type.a.a6c3: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_4.0c1b, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2ecd: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.7cc1 = struct_value (%int_4.2ecd) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %x.var: ref %struct_type.a.7cc1 = var x
// CHECK:STDOUT:   %x: ref %struct_type.a.7cc1 = bind_name x, %x.var
// CHECK:STDOUT:   %y.var: ref %i32 = var y
// CHECK:STDOUT:   %y: ref %i32 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1b]
// CHECK:STDOUT:   %.loc11_27.1: %struct_type.a.a6c3 = struct_literal (%int_4)
// CHECK:STDOUT:   %impl.elem0: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_4, %impl.elem0 [template = constants.%Convert.bound]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_4) [template = constants.%int_4.2ecd]
// CHECK:STDOUT:   %.loc11_27.2: init %i32 = converted %int_4, %int.convert_checked [template = constants.%int_4.2ecd]
// CHECK:STDOUT:   %.loc11_27.3: init %struct_type.a.7cc1 = struct_init (%.loc11_27.2) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_28: init %struct_type.a.7cc1 = converted %.loc11_27.1, %.loc11_27.3 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_28
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.7cc1 = name_ref x, file.%x
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
