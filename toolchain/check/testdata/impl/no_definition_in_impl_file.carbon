// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_definition_in_impl_file.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_definition_in_impl_file.carbon

// --- a.carbon

library "a";

interface AInterface {}

impl i32 as AInterface;

// --- a.impl.carbon

impl library "a";

impl i32 as AInterface;

impl i32 as AInterface {}

// --- b.carbon

library "b";

interface BInterface {}

impl i32 as BInterface;

// --- fail_b.impl.carbon

impl library "b";

import library "a";

// CHECK:STDERR: fail_b.impl.carbon:[[@LINE+3]]:1: ERROR: No definition found for declaration in impl file
// CHECK:STDERR: impl i32 as BInterface;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
impl i32 as BInterface;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @AInterface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .AInterface = %AInterface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %AInterface.decl: type = interface_decl @AInterface [template = constants.%.1] {}
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc6_6.2: type = converted %int.make_type_32, %.loc6_6.1 [template = i32]
// CHECK:STDOUT:     %AInterface.ref: type = name_ref AInterface, %AInterface.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AInterface {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as %.1;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- a.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @AInterface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.3: <witness> = interface_witness () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .AInterface = %import_ref.1
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir0, inst+2, loaded [template = constants.%.1]
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.2 = import_ref ir0, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.3: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32.loc4: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_6.1: type = value_of_initializer %int.make_type_32.loc4 [template = i32]
// CHECK:STDOUT:     %.loc4_6.2: type = converted %int.make_type_32.loc4, %.loc4_6.1 [template = i32]
// CHECK:STDOUT:     %AInterface.ref.loc4: type = name_ref AInterface, %import_ref.1 [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.4: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32.loc6: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_6.1: type = value_of_initializer %int.make_type_32.loc6 [template = i32]
// CHECK:STDOUT:     %.loc6_6.2: type = converted %int.make_type_32.loc6, %.loc6_6.1 [template = i32]
// CHECK:STDOUT:     %AInterface.ref.loc6: type = name_ref AInterface, %import_ref.1 [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AInterface {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.2
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as %.1 {
// CHECK:STDOUT:   %.1: <witness> = interface_witness () [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = %.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @BInterface [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .BInterface = %BInterface.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %BInterface.decl: type = interface_decl @BInterface [template = constants.%.1] {}
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   impl_decl @impl {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc6_6.2: type = converted %int.make_type_32, %.loc6_6.1 [template = i32]
// CHECK:STDOUT:     %BInterface.ref: type = name_ref BInterface, %BInterface.decl [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BInterface {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: i32 as %.1;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @BInterface [template]
// CHECK:STDOUT:   %Self.1: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %.2: type = interface_type @AInterface [template]
// CHECK:STDOUT:   %Self.2: %.2 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .BInterface = %import_ref.1
// CHECK:STDOUT:     .AInterface = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref ir0, inst+2, loaded [template = constants.%.1]
// CHECK:STDOUT:   %import_ref.2 = import_ref ir1, inst+2, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.3 = import_ref ir0, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref ir1, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.5: %Int32.type = import_ref ir4, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   impl_decl @impl.1 {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc9_6.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc9_6.2: type = converted %int.make_type_32, %.loc9_6.1 [template = i32]
// CHECK:STDOUT:     %BInterface.ref: type = name_ref BInterface, %import_ref.1 [template = constants.%.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BInterface {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.3
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @AInterface {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = file.%import_ref.4
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: i32 as %.1;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: i32 as %.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
