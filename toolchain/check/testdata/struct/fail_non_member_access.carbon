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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: %i32} [template]
// CHECK:STDOUT:   %int.4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %int.4.1, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%int.4.2) [template]
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
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int.32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_signed.loc11, %.loc11_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_16: type = struct_type {.a: %i32} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: ref %.1 = bind_name x, %x.var
// CHECK:STDOUT:   %int.32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int.32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_8.2: type = converted %int.make_type_signed.loc15, %.loc15_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %y.var: ref %i32 = var y
// CHECK:STDOUT:   %y: ref %i32 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.4: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc11_27.1: %.2 = struct_literal (%int.4)
// CHECK:STDOUT:   %.loc11_27.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_27.3: <bound method> = bound_method %int.4, %.loc11_27.2 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_27.4: <specific function> = specific_function %.loc11_27.3, @Convert.2(constants.%int.32) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc11_27.4(%int.4) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc11_27.5: init %i32 = converted %int.4, %int.convert_checked [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc11_27.6: init %.1 = struct_init (%.loc11_27.5) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_28: init %.1 = converted %.loc11_27.1, %.loc11_27.6 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_28
// CHECK:STDOUT:   %x.ref: ref %.1 = name_ref x, file.%x
// CHECK:STDOUT:   assign file.%y.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
