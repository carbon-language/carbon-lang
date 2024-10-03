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

// CHECK:STDERR: fail_todo_impl_assoc_const.carbon:[[@LINE+7]]:1: error: semantics TODO: `impl of interface with associated constant`
// CHECK:STDERR: impl bool as I where .T = bool {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_impl_assoc_const.carbon:[[@LINE+3]]:22: error: cannot access member of interface `I` in type `I` that does not implement that interface
// CHECK:STDERR: impl bool as I where .T = bool {}
// CHECK:STDERR:                      ^~
impl bool as I where .T = bool {}

// CHECK:STDOUT: --- fail_todo_impl_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @I [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.2: type = assoc_entity_type %.1, type [template]
// CHECK:STDOUT:   %.3: %.2 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%.1] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %bool.make_type.loc20_6: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc20_6.1: type = value_of_initializer %bool.make_type.loc20_6 [template = bool]
// CHECK:STDOUT:     %.loc20_6.2: type = converted %bool.make_type.loc20_6, %.loc20_6.1 [template = bool]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%.1]
// CHECK:STDOUT:     %.Self: %.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %.1 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %.2 = name_ref T, @I.%.loc11 [template = constants.%.3]
// CHECK:STDOUT:     %bool.make_type.loc20_27: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc20_16: type = where_expr %.Self [template = constants.%.1] {
// CHECK:STDOUT:       requirement_rewrite <error>, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %.loc11: %.2 = assoc_entity element0, %T [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %.loc11
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: bool as %.1 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
