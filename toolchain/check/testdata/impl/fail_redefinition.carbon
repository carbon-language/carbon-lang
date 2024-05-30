// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_redefinition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/impl/fail_redefinition.carbon

interface I {}

impl i32 as I {}

// CHECK:STDERR: fail_redefinition.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `impl i32 as I`.
// CHECK:STDERR: impl i32 as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefinition.carbon:[[@LINE-5]]:1: Previous definition was here.
// CHECK:STDERR: impl i32 as I {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~
impl i32 as I {}

// CHECK:STDOUT: --- fail_redefinition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:     %.loc13_6.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_6.2: type = converted %int.make_type_32.loc13, %.loc13_6.1 [template = i32]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:     %.loc21_6.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_6.2: type = converted %int.make_type_32.loc21, %.loc21_6.1 [template = i32]
// CHECK:STDOUT:     %I.ref.loc21: type = name_ref I, %I.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: I = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as I {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <unexpected instref inst+17>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
