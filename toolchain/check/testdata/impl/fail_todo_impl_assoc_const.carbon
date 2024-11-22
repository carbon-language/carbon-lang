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

// CHECK:STDERR: fail_todo_impl_assoc_const.carbon:[[@LINE+3]]:1: error: semantics TODO: `impl of interface with associated constant` [SemanticsTodo]
// CHECK:STDERR: impl bool as I where .T = bool {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl bool as I where .T = bool {}

// CHECK:STDOUT: --- fail_todo_impl_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.1: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %I.type.1, type [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %.Self: %I.type.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.3: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I where TODO> [template]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type.1] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %bool.make_type.loc16_6: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc16_6.1: type = value_of_initializer %bool.make_type.loc16_6 [template = bool]
// CHECK:STDOUT:     %.loc16_6.2: type = converted %bool.make_type.loc16_6, %.loc16_6.1 [template = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type.1]
// CHECK:STDOUT:     %.Self: %I.type.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type.1 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %.1 = name_ref T, @I.%.loc11 [template = constants.%.2]
// CHECK:STDOUT:     %.loc16_22.1: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.3]
// CHECK:STDOUT:     %.loc16_22.2: type = interface_witness_access %.loc16_22.1, element0
// CHECK:STDOUT:     %bool.make_type.loc16_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc16_27.1: type = value_of_initializer %bool.make_type.loc16_27 [template = bool]
// CHECK:STDOUT:     %.loc16_27.2: type = converted %bool.make_type.loc16_27, %.loc16_27.1 [template = bool]
// CHECK:STDOUT:     %.loc16_16: type = where_expr %.Self [template = constants.%I.type.2] {
// CHECK:STDOUT:       requirement_rewrite %.loc16_22.2, %.loc16_27.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc11: %.1 = assoc_entity element0, %T [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc11
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc16_6.2 as %.loc16_16 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
