// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_impl_bad_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_impl_bad_interface.carbon

// --- fail_impl_as_false.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_impl_as_false.carbon:[[@LINE+7]]:13: error: cannot implicitly convert from `bool` to `type` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl i32 as false {}
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR: fail_impl_as_false.carbon:[[@LINE+4]]:13: note: type `bool` does not implement interface `ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl i32 as false {}
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR:
impl i32 as false {}

// --- fail_impl_as_type.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_impl_as_type.carbon:[[@LINE+4]]:1: error: impl as non-facet-type [ImplAsNonFacetType]
// CHECK:STDERR: impl bool as type {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl bool as type {}

// --- fail_impl_as_type_where.carbon

library "[[@TEST_NAME]]";

// Note: This code is not expected to pass checking, even once the TODO is
// addressed.

// CHECK:STDERR: fail_impl_as_type_where.carbon:[[@LINE+3]]:1: error: semantics TODO: `impl as not 1 interface` [SemanticsTodo]
// CHECK:STDERR: impl f64 as type where .Self impls type {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl f64 as type where .Self impls type {}

// CHECK:STDOUT: --- fail_impl_as_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11: type = converted @impl.9.%.loc11_13, <error> [template = <error>]
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %.loc11_6.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc11_6.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_6.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_6.3: type = converted %int.make_type_signed, %.loc11_6.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_13: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %.loc11_6.3 as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_6.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc8_6.2: type = converted %bool.make_type, %.loc8_6.1 [template = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_6.2 as type {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_as_type_where.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.Self: type = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where TODO> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc10_6.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%.loc10_6.1) [template = f64]
// CHECK:STDOUT:     %.loc10_6.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:     %.loc10_6.3: type = converted %float.make_type, %.loc10_6.2 [template = f64]
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.loc10_18: type = where_expr %.Self [template = constants.%type_where] {
// CHECK:STDOUT:       requirement_impls %.Self.ref, type
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc10_6.3 as %.loc10_18 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
