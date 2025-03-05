// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_non_tuple_access.carbon

fn Main() {
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:3: error: type `Core.IntLiteral` does not support indexing [TypeNotIndexable]
  // CHECK:STDERR:   0[1];
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  0[1];

  var non_tuple: array(i32, 2) = (5, 5);
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:20: error: type `[i32; 2]` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   var first: i32 = non_tuple.0;
  // CHECK:STDERR:                    ^~~~~~~~~~~
  // CHECK:STDERR:
  var first: i32 = non_tuple.0;
}

// CHECK:STDOUT: --- fail_non_tuple_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_5.0f6, %int_5.0f6) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IndexWith = %Core.IndexWith
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc16: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %non_tuple.patt: %array_type = binding_pattern non_tuple
// CHECK:STDOUT:     %.loc18_3.1: %array_type = var_pattern %non_tuple.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_tuple.var: ref %array_type = var non_tuple
// CHECK:STDOUT:   %int_5.loc18_35: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_5.loc18_38: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc18_39.1: %tuple.type = tuple_literal (%int_5.loc18_35, %int_5.loc18_38)
// CHECK:STDOUT:   %impl.elem0.loc18_39.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_39.1: <bound method> = bound_method %int_5.loc18_35, %impl.elem0.loc18_39.1 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc18_39.1: <specific function> = specific_function %impl.elem0.loc18_39.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_39.2: <bound method> = bound_method %int_5.loc18_35, %specific_fn.loc18_39.1 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc18_39.1: init %i32 = call %bound_method.loc18_39.2(%int_5.loc18_35) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc18_39.2: init %i32 = converted %int_5.loc18_35, %int.convert_checked.loc18_39.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %int_0.loc18: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc18_39.3: ref %i32 = array_index %non_tuple.var, %int_0.loc18
// CHECK:STDOUT:   %.loc18_39.4: init %i32 = initialize_from %.loc18_39.2 to %.loc18_39.3 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc18_39.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_39.3: <bound method> = bound_method %int_5.loc18_38, %impl.elem0.loc18_39.2 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc18_39.2: <specific function> = specific_function %impl.elem0.loc18_39.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_39.4: <bound method> = bound_method %int_5.loc18_38, %specific_fn.loc18_39.2 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked.loc18_39.2: init %i32 = call %bound_method.loc18_39.4(%int_5.loc18_38) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc18_39.5: init %i32 = converted %int_5.loc18_38, %int.convert_checked.loc18_39.2 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc18_39.6: ref %i32 = array_index %non_tuple.var, %int_1.loc18
// CHECK:STDOUT:   %.loc18_39.7: init %i32 = initialize_from %.loc18_39.5 to %.loc18_39.6 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc18_39.8: init %array_type = array_init (%.loc18_39.4, %.loc18_39.7) to %non_tuple.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc18_3.2: init %array_type = converted %.loc18_39.1, %.loc18_39.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %non_tuple.var, %.loc18_3.2
// CHECK:STDOUT:   %.loc18_30: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %non_tuple: ref %array_type = bind_name non_tuple, %non_tuple.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %first.patt: %i32 = binding_pattern first
// CHECK:STDOUT:     %.loc23_3: %i32 = var_pattern %first.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %first.var: ref %i32 = var first
// CHECK:STDOUT:   %non_tuple.ref: ref %array_type = name_ref non_tuple, %non_tuple
// CHECK:STDOUT:   %int_0.loc23: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   assign %first.var, <error>
// CHECK:STDOUT:   %.loc23_14: type = splice_block %i32.loc23 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %first: ref %i32 = bind_name first, %first.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
