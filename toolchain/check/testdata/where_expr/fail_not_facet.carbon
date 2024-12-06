// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/fail_not_facet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/fail_not_facet.carbon

// --- fail_left_where_not_facet.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_left_where_not_facet.carbon:[[@LINE+4]]:10: error: left argument of `where` operator must be a facet type [WhereOnNonFacetType]
// CHECK:STDERR: fn F(T:! i32 where .Self == bool);
// CHECK:STDERR:          ^~~
// CHECK:STDERR:
fn F(T:! i32 where .Self == bool);

// --- fail_left_where_unknown.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_left_where_unknown.carbon:[[@LINE+4]]:10: error: name `NOT_DECLARED` not found [NameNotFound]
// CHECK:STDERR: fn G(U:! NOT_DECLARED where .Self == bool);
// CHECK:STDERR:          ^~~~~~~~~~~~
// CHECK:STDERR:
fn G(U:! NOT_DECLARED where .Self == bool);

// --- fail_var.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_var.carbon:[[@LINE+3]]:8: error: name `e` not found [NameNotFound]
// CHECK:STDERR: var v: e where .x = 3;
// CHECK:STDERR:        ^
var v: e where .x = 3;

// CHECK:STDOUT: --- fail_left_where_not_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %T.patt: <error> = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_6.1: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: <error> = value_param_pattern %T.patt.loc8_6.1, runtime_param<invalid> [symbolic = %T.patt.loc8_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_10.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_10.2: type = converted %int.make_type_signed, %.loc8_10.1 [template = constants.%i32]
// CHECK:STDOUT:     %.Self: <error> = bind_symbolic_name .Self [template = <error>]
// CHECK:STDOUT:     %.Self.ref: <error> = name_ref .Self, %.Self [template = <error>]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %bool.make_type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.param: <error> = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T: <error> = bind_symbolic_name T, 0, %T.param [template = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T: <error>) {
// CHECK:STDOUT:   %T.patt.loc8_6.2: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(<error>) {
// CHECK:STDOUT:   %T.patt.loc8_6.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_left_where_unknown.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %U.patt: <error> = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %U.patt.loc8_6.1: <error> = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: <error> = value_param_pattern %U.patt.loc8_6.1, runtime_param<invalid> [symbolic = %U.patt.loc8_6.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NOT_DECLARED.ref: <error> = name_ref NOT_DECLARED, <error> [template = <error>]
// CHECK:STDOUT:     %.Self: <error> = bind_symbolic_name .Self [template = <error>]
// CHECK:STDOUT:     %.Self.ref: <error> = name_ref .Self, %.Self [template = <error>]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %bool.make_type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: <error> = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U: <error> = bind_symbolic_name U, 0, %U.param [template = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U: <error>) {
// CHECK:STDOUT:   %U.patt.loc8_6.2: <error> = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_6.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(<error>) {
// CHECK:STDOUT:   %U.patt.loc8_6.2 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %e.ref: <error> = name_ref e, <error> [template = <error>]
// CHECK:STDOUT:   %.Self: <error> = bind_symbolic_name .Self [template = <error>]
// CHECK:STDOUT:   %.Self.ref: <error> = name_ref .Self, %.Self [template = <error>]
// CHECK:STDOUT:   %x.ref: <error> = name_ref x, <error> [template = <error>]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc7: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:     requirement_rewrite %x.ref, <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
