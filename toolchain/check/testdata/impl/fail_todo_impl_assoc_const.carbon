// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_todo_impl_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_todo_impl_assoc_const.carbon

interface I { let T:! type; }

// CHECK:STDERR: fail_todo_impl_assoc_const.carbon:[[@LINE+10]]:1: error: semantics TODO: `impl of interface with associated constant` [SemanticsTodo]
// CHECK:STDERR: impl bool as I where .T = bool {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_impl_assoc_const.carbon:[[@LINE+6]]:27: error: cannot implicitly convert from `type` to `<associated type in I>` [ImplicitAsConversionFailure]
// CHECK:STDERR: impl bool as I where .T = bool {}
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR: fail_todo_impl_assoc_const.carbon:[[@LINE+3]]:27: note: type `type` does not implement interface `ImplicitAs(<associated type in I>)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: impl bool as I where .T = bool {}
// CHECK:STDERR:                           ^~~~
impl bool as I where .T = bool {}

// CHECK:STDOUT: --- fail_todo_impl_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.1: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %I.type.1, type [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %.Self: %I.type.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I where TODO> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type.1] {} {}
// CHECK:STDOUT:   impl_decl @impl.9 [template] {} {
// CHECK:STDOUT:     %bool.make_type.loc23_6: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc23_6.1: type = value_of_initializer %bool.make_type.loc23_6 [template = bool]
// CHECK:STDOUT:     %.loc23_6.2: type = converted %bool.make_type.loc23_6, %.loc23_6.1 [template = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type.1]
// CHECK:STDOUT:     %.Self: %I.type.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type.1 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %.1 = name_ref T, @I.%.loc11 [template = constants.%.2]
// CHECK:STDOUT:     %bool.make_type.loc23_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc23_27: %.1 = converted %bool.make_type.loc23_27, <error> [template = <error>]
// CHECK:STDOUT:     %.loc23_16: type = where_expr %.Self [template = constants.%I.type.2] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc11: %.1 = assoc_entity element0, %T [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc11
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.9: %.loc23_6.2 as %.loc23_16 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
