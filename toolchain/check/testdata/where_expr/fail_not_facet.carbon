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

// CHECK:STDERR: fail_var.carbon:[[@LINE+4]]:8: error: name `e` not found [NameNotFound]
// CHECK:STDERR: var v: e where .x = 3;
// CHECK:STDERR:        ^
// CHECK:STDERR:
var v: e where .x = 3;

// CHECK:STDOUT: --- fail_left_where_not_facet.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: <error> = symbolic_binding_pattern T, 0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [concrete = <error>] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %.Self: <error> = bind_symbolic_name .Self [concrete = <error>]
// CHECK:STDOUT:       %.Self.ref: <error> = name_ref .Self, %.Self [concrete = <error>]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc8_14.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %bool.make_type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T: <error> = bind_symbolic_name T, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T: <error>) {
// CHECK:STDOUT:   fn(%T.patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_left_where_unknown.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .NOT_DECLARED = <poisoned>
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt: <error> = symbolic_binding_pattern U, 0 [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_23.1: type = splice_block %.loc8_23.2 [concrete = <error>] {
// CHECK:STDOUT:       %NOT_DECLARED.ref: <error> = name_ref NOT_DECLARED, <error> [concrete = <error>]
// CHECK:STDOUT:       %.Self: <error> = bind_symbolic_name .Self [concrete = <error>]
// CHECK:STDOUT:       %.Self.ref: <error> = name_ref .Self, %.Self [concrete = <error>]
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc8_23.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:         requirement_equivalent %.Self.ref, %bool.make_type
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U: <error> = bind_symbolic_name U, 0 [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U: <error>) {
// CHECK:STDOUT:   fn(%U.patt: <error>);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(<error>) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .e = <poisoned>
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v
// CHECK:STDOUT:     %.loc8_1: <error> = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %.loc8_10.1: type = splice_block %.loc8_10.2 [concrete = <error>] {
// CHECK:STDOUT:     %e.ref: <error> = name_ref e, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self: <error> = bind_symbolic_name .Self [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref: <error> = name_ref .Self, %.Self [concrete = <error>]
// CHECK:STDOUT:     %x.ref: <error> = name_ref x, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc8_10.2: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %x.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: <error> = bind_name v, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
