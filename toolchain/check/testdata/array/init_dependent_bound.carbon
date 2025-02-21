// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/init_dependent_bound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/init_dependent_bound.carbon

// --- fail_init_dependent_bound.carbon

library "[[@TEST_NAME]]";

fn F(N:! i32) {
  // CHECK:STDERR: fail_init_dependent_bound.carbon:[[@LINE+4]]:28: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                            ^~~~~~~~~
  // CHECK:STDERR:
  var arr: array(i32, N) = (1, 2, 3);
}

// --- fail_todo_init_template_dependent_bound.carbon

library "[[@TEST_NAME]]";

// TODO: This should be valid.
fn G(template N:! i32) {
  // CHECK:STDERR: fail_todo_init_template_dependent_bound.carbon:[[@LINE+4]]:28: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var arr: array(i32, N) = (1, 2, 3);
  // CHECK:STDERR:                            ^~~~~~~~~
  // CHECK:STDERR:
  var arr: array(i32, N) = (1, 2, 3);
}

fn H() { G(3); }

// CHECK:STDOUT: --- fail_init_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2fd = facet_value %i32, %impl_witness.023 [concrete]
// CHECK:STDOUT:   %.10e: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %int.convert_checked, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.9dc: <witness> = require_complete_type %array_type [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:     %N.patt.loc4_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_6.1, runtime_param<none> [symbolic = %N.patt.loc4_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc4: type = splice_block %i32.loc4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc4_6.1: %i32) {
// CHECK:STDOUT:   %N.loc4_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc4_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc4_6.2, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn)]
// CHECK:STDOUT:   %int.convert_checked.loc9_23.2: init Core.IntLiteral = call %Convert.specific_fn(%N.loc4_6.2) [symbolic = %int.convert_checked.loc9_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc9_24.2: type = array_type %int.convert_checked.loc9_23.2, %i32 [symbolic = %array_type.loc9_24.2 (constants.%array_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc9_24.2 (%array_type) [symbolic = %require_complete (constants.%require_complete.9dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @F.%array_type.loc9_24.2 (%array_type) = binding_pattern arr
// CHECK:STDOUT:       %.loc9_3: @F.%array_type.loc9_24.2 (%array_type) = var_pattern %arr.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @F.%array_type.loc9_24.2 (%array_type) = var arr
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %.loc9_36: %tuple.type = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:     assign %arr.var, <error>
// CHECK:STDOUT:     %.loc9_24: type = splice_block %array_type.loc9_24.1 [symbolic = %array_type.loc9_24.2 (constants.%array_type)] {
// CHECK:STDOUT:       %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc4_6.1 [symbolic = %N.loc4_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.10e = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %bound_method, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn)]
// CHECK:STDOUT:       %int.convert_checked.loc9_23.1: init Core.IntLiteral = call %specific_fn(%N.ref) [symbolic = %int.convert_checked.loc9_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc9_23.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc9_23.1 [symbolic = %int.convert_checked.loc9_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc9_23.2: Core.IntLiteral = converted %N.ref, %.loc9_23.1 [symbolic = %int.convert_checked.loc9_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc9_24.1: type = array_type %.loc9_23.2, %i32 [symbolic = %array_type.loc9_24.2 (constants.%array_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @F.%array_type.loc9_24.2 (%array_type) = bind_name arr, %arr.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc4_6.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc4_6.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_init_template_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0, template [template]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0, template [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.2(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e25: %ImplicitAs.type.2fd = facet_value %i32, %impl_witness.023 [concrete]
// CHECK:STDOUT:   %.10e: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.e25 [concrete]
// CHECK:STDOUT:   %Convert.bound.588: <bound method> = bound_method %N.51e, %Convert.960 [template]
// CHECK:STDOUT:   %Convert.specific_fn.18b: <specific function> = specific_function %Convert.bound.588, @Convert.3(%int_32) [template]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.18b(%N.51e) [template]
// CHECK:STDOUT:   %array_type.b04: type = array_type %int.convert_checked, %i32 [template]
// CHECK:STDOUT:   %require_complete.9dc: <witness> = require_complete_type %array_type.b04 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f7f: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.f7f [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b42: <specific function> = specific_function %Convert.bound.b30, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%int_3.822) [concrete]
// CHECK:STDOUT:   %Convert.bound.2d6: <bound method> = bound_method %int_3.822, %Convert.960 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.377: <specific function> = specific_function %Convert.bound.2d6, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %array_type.3fb: type = array_type %int_3.1ba, %i32 [concrete]
// CHECK:STDOUT:   %complete_type.cbf: <witness> = complete_type_witness %array_type.3fb [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %N.patt.loc5_15.1: %i32 = symbolic_binding_pattern N, 0, template [template = %N.patt.loc5_15.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc5_15.1, runtime_param<none> [template = %N.patt.loc5_15.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc5: type = splice_block %i32.loc5 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc5_15.1: %i32 = bind_symbolic_name N, 0, template, %N.param [template = %N.loc5_15.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%N.loc5_15.1: %i32) {
// CHECK:STDOUT:   %N.loc5_15.2: %i32 = bind_symbolic_name N, 0, template [template = %N.loc5_15.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc5_15.2: %i32 = symbolic_binding_pattern N, 0, template [template = %N.patt.loc5_15.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc5_15.2, constants.%Convert.960 [template = %Convert.bound (constants.%Convert.bound.588)]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(constants.%int_32) [template = %Convert.specific_fn (constants.%Convert.specific_fn.18b)]
// CHECK:STDOUT:   %int.convert_checked.loc10_23.2: init Core.IntLiteral = call %Convert.specific_fn(%N.loc5_15.2) [template = %int.convert_checked.loc10_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc10_24.2: type = array_type %int.convert_checked.loc10_23.2, %i32 [template = %array_type.loc10_24.2 (constants.%array_type.b04)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%array_type.loc10_24.2 (%array_type.b04) [template = %require_complete (constants.%require_complete.9dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @G.%array_type.loc10_24.2 (%array_type.b04) = binding_pattern arr
// CHECK:STDOUT:       %.loc10_3: @G.%array_type.loc10_24.2 (%array_type.b04) = var_pattern %arr.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @G.%array_type.loc10_24.2 (%array_type.b04) = var arr
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc10_36: %tuple.type = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:     assign %arr.var, <error>
// CHECK:STDOUT:     %.loc10_24: type = splice_block %array_type.loc10_24.1 [template = %array_type.loc10_24.2 (constants.%array_type.b04)] {
// CHECK:STDOUT:       %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc5_15.1 [template = %N.loc5_15.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.10e = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %N.ref, %impl.elem0 [template = %Convert.bound (constants.%Convert.bound.588)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %bound_method, @Convert.3(constants.%int_32) [template = %Convert.specific_fn (constants.%Convert.specific_fn.18b)]
// CHECK:STDOUT:       %int.convert_checked.loc10_23.1: init Core.IntLiteral = call %specific_fn(%N.ref) [template = %int.convert_checked.loc10_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc10_23.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc10_23.1 [template = %int.convert_checked.loc10_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc10_23.2: Core.IntLiteral = converted %N.ref, %.loc10_23.1 [template = %int.convert_checked.loc10_23.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc10_24.1: type = array_type %.loc10_23.2, %i32 [template = %array_type.loc10_24.2 (constants.%array_type.b04)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @G.%array_type.loc10_24.2 (%array_type.b04) = bind_name arr, %arr.var
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_3, %impl.elem0 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %bound_method, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b42]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %specific_fn(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_13.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc13_13.2: %i32 = converted %int_3, %.loc13_13.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%int_3.822) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc5_15.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc5_15.2 => constants.%N.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%int_3.822) {
// CHECK:STDOUT:   %N.loc5_15.2 => constants.%int_3.822
// CHECK:STDOUT:   %N.patt.loc5_15.2 => constants.%int_3.822
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.2d6
// CHECK:STDOUT:   %Convert.specific_fn => constants.%Convert.specific_fn.377
// CHECK:STDOUT:   %int.convert_checked.loc10_23.2 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc10_24.2 => constants.%array_type.3fb
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cbf
// CHECK:STDOUT: }
// CHECK:STDOUT:
