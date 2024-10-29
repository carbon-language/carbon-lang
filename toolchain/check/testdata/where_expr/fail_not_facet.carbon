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

// CHECK:STDERR: fail_left_where_unknown.carbon:[[@LINE+8]]:10: error: name `NOT_DECLARED` not found [NameNotFound]
// CHECK:STDERR: fn G(U:! NOT_DECLARED where .Self == bool);
// CHECK:STDERR:          ^~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_left_where_unknown.carbon:[[@LINE+4]]:10: error: left argument of `where` operator must be a facet type [WhereOnNonFacetType]
// CHECK:STDERR: fn G(U:! NOT_DECLARED where .Self == bool);
// CHECK:STDERR:          ^~~~~~~~~~~~
// CHECK:STDERR:
fn G(U:! NOT_DECLARED where .Self == bool);

// --- fail_var.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_var.carbon:[[@LINE+7]]:8: error: name `e` not found [NameNotFound]
// CHECK:STDERR: var v: e where .x = 3;
// CHECK:STDERR:        ^
// CHECK:STDERR:
// CHECK:STDERR: fail_var.carbon:[[@LINE+3]]:8: error: left argument of `where` operator must be a facet type [WhereOnNonFacetType]
// CHECK:STDERR: var v: e where .x = 3;
// CHECK:STDERR:        ^
var v: e where .x = 3;

// CHECK:STDOUT: --- fail_left_where_not_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.Self: <error> = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %T: <error> = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: <error> = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Bool.type = import_ref Core//prelude/types/bool, inst+5, loaded [template = constants.%Bool]
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
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_10.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_10.2: type = converted %int.make_type_32, %.loc8_10.1 [template = i32]
// CHECK:STDOUT:     %.Self: <error> = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: <error> = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %bool.make_type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.param: <error> = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_6.1: <error> = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_6.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_6.1: <error>) {
// CHECK:STDOUT:   %T.loc8_6.2: <error> = bind_symbolic_name T, 0 [symbolic = %T.loc8_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_6.2: <error> = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc8_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_left_where_unknown.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: <error> = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %U: <error> = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: <error> = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref: %Bool.type = import_ref Core//prelude/types/bool, inst+5, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %U.patt.loc12_6.1: <error> = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: <error> = value_param_pattern %U.patt.loc12_6.1, runtime_param<invalid> [symbolic = %U.patt.loc12_6.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %NOT_DECLARED.ref: <error> = name_ref NOT_DECLARED, <error> [template = <error>]
// CHECK:STDOUT:     %.Self: <error> = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: <error> = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc12: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_equivalent %.Self.ref, %bool.make_type
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: <error> = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc12_6.1: <error> = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc12_6.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U.loc12_6.1: <error>) {
// CHECK:STDOUT:   %U.loc12_6.2: <error> = bind_symbolic_name U, 0 [symbolic = %U.loc12_6.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc12_6.2: <error> = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_6.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U.loc12_6.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc12_6.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self: <error> = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: i32 = int_literal 3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:   %.Self: <error> = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:   %.Self.ref: <error> = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:   %.loc11_21: i32 = int_literal 3 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_10: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:     requirement_rewrite <error>, <error>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
