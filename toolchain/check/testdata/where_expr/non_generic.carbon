// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/non_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/non_generic.carbon

interface I { let T:! type; }

// Ensure that we don't crash when checking this `where` in a non-generic context.
fn NotGenericF(U: I where .T == i32) {}

// CHECK:STDOUT: --- non_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type.1: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self: %I.type.1 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.1: type = assoc_entity_type %I.type.1, type [template]
// CHECK:STDOUT:   %.2: %.1 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %.Self: %I.type.1 = bind_symbolic_name .Self, 0 [symbolic]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.3 [template]
// CHECK:STDOUT:   %I.type.2: type = facet_type <@I where TODO> [template]
// CHECK:STDOUT:   %NotGenericF.type: type = fn_type @NotGenericF [template]
// CHECK:STDOUT:   %NotGenericF: %NotGenericF.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .NotGenericF = %NotGenericF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type.1] {} {}
// CHECK:STDOUT:   %NotGenericF.decl: %NotGenericF.type = fn_decl @NotGenericF [template = constants.%NotGenericF] {
// CHECK:STDOUT:     %U.patt: %I.type.2 = binding_pattern U
// CHECK:STDOUT:     %U.param_patt: %I.type.2 = value_param_pattern %U.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type.1]
// CHECK:STDOUT:     %.Self: %I.type.1 = bind_symbolic_name .Self, 0 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type.1 = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %.1 = name_ref T, @I.%.loc11 [template = constants.%.2]
// CHECK:STDOUT:     %.loc14_33: Core.IntLiteral = int_value 32 [template = constants.%.3]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc14_33) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_21: type = where_expr %.Self [template = constants.%I.type.2] {
// CHECK:STDOUT:       requirement_equivalent %T.ref, %int.make_type_signed
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.param: %I.type.2 = value_param runtime_param0
// CHECK:STDOUT:     %U: %I.type.2 = bind_name U, %U.param
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
// CHECK:STDOUT: fn @NotGenericF(%U.param_patt: %I.type.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
