// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/fail_negative_indexing.carbon

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+4]]:14: error: tuple element index `-10` is past the end of type `(i32, i32)` [TupleIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a.(-10);
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
var b: i32 = a.(-10);

// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.221: <bound method> = bound_method %int_12.6a3, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.dae: <bound method> = bound_method %int_12.6a3, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_12.1e1: %i32 = int_value 12 [concrete]
// CHECK:STDOUT:   %Convert.bound.ce9: <bound method> = bound_method %int_6.462, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.efa: <bound method> = bound_method %int_6.462, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_12.1e1, %int_6.e56) [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %impl_witness.0f6: <witness> = impl_witness (imports.%Core.import_ref.c15) [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%impl_witness.0f6) [concrete]
// CHECK:STDOUT:   %.690: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.type.1be: type = fn_type @Op.15 [concrete]
// CHECK:STDOUT:   %Op.bba: %Op.type.1be = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_10, %Op.bba [concrete]
// CHECK:STDOUT:   %int_-10: Core.IntLiteral = int_value -10 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %tuple.type.d07 = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %tuple.type.d07 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.d07 = var a
// CHECK:STDOUT:   %.loc11_17.1: type = splice_block %.loc11_17.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc11_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_17.2: %tuple.type.24b = tuple_literal (%i32.loc11_9, %i32.loc11_14)
// CHECK:STDOUT:     %.loc11_17.3: type = converted %.loc11_17.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.d07 = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc16_1: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %.loc16_8: type = splice_block %i32.loc16 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %.loc11_27.1: %tuple.type.f94 = tuple_literal (%int_12, %int_6)
// CHECK:STDOUT:   %impl.elem0.loc11_27.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_27.1: <bound method> = bound_method %int_12, %impl.elem0.loc11_27.1 [concrete = constants.%Convert.bound.221]
// CHECK:STDOUT:   %specific_fn.loc11_27.1: <specific function> = specific_function %impl.elem0.loc11_27.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_27.2: <bound method> = bound_method %int_12, %specific_fn.loc11_27.1 [concrete = constants.%bound_method.dae]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.1: init %i32 = call %bound_method.loc11_27.2(%int_12) [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc11_27.2: init %i32 = converted %int_12, %int.convert_checked.loc11_27.1 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_27.3: init %i32 = initialize_from %.loc11_27.2 to %tuple.elem0 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %impl.elem0.loc11_27.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_27.3: <bound method> = bound_method %int_6, %impl.elem0.loc11_27.2 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:   %specific_fn.loc11_27.2: <specific function> = specific_function %impl.elem0.loc11_27.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_27.4: <bound method> = bound_method %int_6, %specific_fn.loc11_27.2 [concrete = constants.%bound_method.efa]
// CHECK:STDOUT:   %int.convert_checked.loc11_27.2: init %i32 = call %bound_method.loc11_27.4(%int_6) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc11_27.4: init %i32 = converted %int_6, %int.convert_checked.loc11_27.2 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_27.5: init %i32 = initialize_from %.loc11_27.4 to %tuple.elem1 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc11_27.6: init %tuple.type.d07 = tuple_init (%.loc11_27.3, %.loc11_27.5) to file.%a.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_1: init %tuple.type.d07 = converted %.loc11_27.1, %.loc11_27.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   %a.ref: ref %tuple.type.d07 = name_ref a, file.%a
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %impl.elem0.loc16: %.690 = impl_witness_access constants.%impl_witness.0f6, element0 [concrete = constants.%Op.bba]
// CHECK:STDOUT:   %bound_method.loc16: <bound method> = bound_method %int_10, %impl.elem0.loc16 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %bound_method.loc16(%int_10) [concrete = constants.%int_-10]
// CHECK:STDOUT:   %.loc16_17.1: Core.IntLiteral = value_of_initializer %int.snegate [concrete = constants.%int_-10]
// CHECK:STDOUT:   %.loc16_17.2: Core.IntLiteral = converted %int.snegate, %.loc16_17.1 [concrete = constants.%int_-10]
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
