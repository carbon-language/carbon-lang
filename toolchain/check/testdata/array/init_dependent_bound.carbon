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
  // CHECK:STDERR: fail_init_dependent_bound.carbon:[[@LINE+4]]:23: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var arr: [i32; N] = (1, 2, 3);
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR:
  var arr: [i32; N] = (1, 2, 3);
}

// --- fail_todo_init_template_dependent_bound.carbon

library "[[@TEST_NAME]]";

// TODO: This should be valid.
// CHECK:STDERR: fail_todo_init_template_dependent_bound.carbon:[[@LINE+4]]:6: error: semantics TODO: `HandleTemplate` [SemanticsTodo]
// CHECK:STDERR: fn G(template N:! i32) {
// CHECK:STDERR:      ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(template N:! i32) {
  var arr: [i32; N] = (1, 2, 3);
}

fn H() { G(3); }

// CHECK:STDOUT: --- fail_init_dependent_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [template]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.2(%int_32) [template]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2fd = facet_value %i32, %impl_witness.023 [template]
// CHECK:STDOUT:   %.10e: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %int.convert_checked, %i32 [symbolic]
// CHECK:STDOUT:   %require_complete.9dc: <witness> = require_complete_type %array_type [symbolic]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
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
// CHECK:STDOUT:     %N.patt.loc4_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_6.1, runtime_param<none> [symbolic = %N.patt.loc4_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<none>
// CHECK:STDOUT:     %.loc4: type = splice_block %i32.loc4 [template = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc4: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc4: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
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
// CHECK:STDOUT:   %int.convert_checked.loc9_18.2: init Core.IntLiteral = call %Convert.specific_fn(%N.loc4_6.2) [symbolic = %int.convert_checked.loc9_18.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc9_19.2: type = array_type %int.convert_checked.loc9_18.2, %i32 [symbolic = %array_type.loc9_19.2 (constants.%array_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc9_19.2 (%array_type) [symbolic = %require_complete (constants.%require_complete.9dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%N.param_patt: %i32) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %arr.patt: @F.%array_type.loc9_19.2 (%array_type) = binding_pattern arr
// CHECK:STDOUT:       %.loc9_3: @F.%array_type.loc9_19.2 (%array_type) = var_pattern %arr.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr.var: ref @F.%array_type.loc9_19.2 (%array_type) = var arr
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:     %.loc9_31: %tuple.type = tuple_literal (%int_1, %int_2, %int_3)
// CHECK:STDOUT:     assign %arr.var, <error>
// CHECK:STDOUT:     %.loc9_19: type = splice_block %array_type.loc9_19.1 [symbolic = %array_type.loc9_19.2 (constants.%array_type)] {
// CHECK:STDOUT:       %int_32.loc9: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:       %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc4_6.1 [symbolic = %N.loc4_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.10e = impl_witness_access constants.%impl_witness.023, element0 [template = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %bound_method, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn)]
// CHECK:STDOUT:       %int.convert_checked.loc9_18.1: init Core.IntLiteral = call %specific_fn(%N.ref) [symbolic = %int.convert_checked.loc9_18.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc9_18.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc9_18.1 [symbolic = %int.convert_checked.loc9_18.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc9_18.2: Core.IntLiteral = converted %N.ref, %.loc9_18.1 [symbolic = %int.convert_checked.loc9_18.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc9_19.1: type = array_type %.loc9_18.2, %i32 [symbolic = %array_type.loc9_19.2 (constants.%array_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %arr: ref @F.%array_type.loc9_19.2 (%array_type) = bind_name arr, %arr.var
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
