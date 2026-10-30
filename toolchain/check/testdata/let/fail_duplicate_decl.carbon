// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/fail_duplicate_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/fail_duplicate_decl.carbon

fn F() {
  let unused a: i32 = 1;
  // CHECK:STDERR:        ╭─┤ fail_duplicate_decl.carbon:[[@LINE+11]]:14
  // CHECK:STDERR:        │   let unused a: i32 = 1;
  // CHECK:STDERR:        ·              ───┬──
  // CHECK:STDERR:        ·                 ╰──┤ name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   let unused a: i32 = 2;
  // CHECK:STDERR:        ·              ┳
  // CHECK:STDERR:        ·              ┗━━┫ error: duplicate name `a` being declared in the same scope
  // CHECK:STDERR:        ·                 ┃        [NameDeclDuplicate]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let unused a: i32 = 2;
}

// CHECK:STDOUT: --- fail_duplicate_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt.4135b0.1: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %a.patt.4135b0.2: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc16_23.1: <bound method> = bound_method %int_1, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_23.2: <bound method> = bound_method %int_1, %specific_fn.loc16 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i32 = call %bound_method.loc16_23.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc16_23.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc16_23.2: %i32 = converted %int_1, %.loc16_23.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a.loc16: %i32 = wrapper_binding a, %.loc16_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt.loc16: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt.4135b0.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc28: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc28_23.1: <bound method> = bound_method %int_2, %impl.elem0.loc28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc28: <specific function> = specific_function %impl.elem0.loc28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_23.2: <bound method> = bound_method %int_2, %specific_fn.loc28 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %i32 = call %bound_method.loc28_23.2(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc28_23.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc28_23.2: %i32 = converted %int_2, %.loc28_23.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %i32.loc28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a.loc28: %i32 = wrapper_binding a, %.loc28_23.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt.loc28: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt.4135b0.2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
