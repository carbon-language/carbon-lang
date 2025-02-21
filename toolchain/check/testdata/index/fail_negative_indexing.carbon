// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon

var c: array(i32, 2) = (42, 42);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+4]]:16: error: array index `-10` is past the end of type `[i32; 2]` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var d: i32 = c[-10];
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
var d: i32 = c[-10];


// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.132: <bound method> = bound_method %int_42.20e, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.2f6: <specific function> = specific_function %Convert.bound.132, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_42.c68, %int_42.c68) [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %impl_witness.0f6: <witness> = impl_witness (imports.%Core.import_ref.c15) [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.13 [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, %impl_witness.0f6 [concrete]
// CHECK:STDOUT:   %.45d: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.type.1be: type = fn_type @Op.14 [concrete]
// CHECK:STDOUT:   %Op.bba: %Op.type.1be = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_10, %Op.bba [concrete]
// CHECK:STDOUT:   %int_-10.06d: Core.IntLiteral = int_value -10 [concrete]
// CHECK:STDOUT:   %Convert.bound.1ca: <bound method> = bound_method %int_-10.06d, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.4be: <specific function> = specific_function %Convert.bound.1ca, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_-10.c17: %i32 = int_value -10 [concrete]
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
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %array_type = binding_pattern c
// CHECK:STDOUT:     %.loc11_1: %array_type = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %array_type = var c
// CHECK:STDOUT:   %.loc11_20: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %array_type = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %i32 = binding_pattern d
// CHECK:STDOUT:     %.loc16_1: %i32 = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %.loc16_8: type = splice_block %i32.loc16 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_42.loc11_25: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %int_42.loc11_29: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type = tuple_literal (%int_42.loc11_25, %int_42.loc11_29)
// CHECK:STDOUT:   %impl.elem0.loc11_31.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_31.1: <bound method> = bound_method %int_42.loc11_25, %impl.elem0.loc11_31.1 [concrete = constants.%Convert.bound.132]
// CHECK:STDOUT:   %specific_fn.loc11_31.1: <specific function> = specific_function %bound_method.loc11_31.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.2f6]
// CHECK:STDOUT:   %int.convert_checked.loc11_31.1: init %i32 = call %specific_fn.loc11_31.1(%int_42.loc11_25) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc11_31.2: init %i32 = converted %int_42.loc11_25, %int.convert_checked.loc11_31.1 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_31.3: ref %i32 = array_index file.%c.var, %int_0
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.2 to %.loc11_31.3 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %impl.elem0.loc11_31.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_31.2: <bound method> = bound_method %int_42.loc11_29, %impl.elem0.loc11_31.2 [concrete = constants.%Convert.bound.132]
// CHECK:STDOUT:   %specific_fn.loc11_31.2: <specific function> = specific_function %bound_method.loc11_31.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.2f6]
// CHECK:STDOUT:   %int.convert_checked.loc11_31.2: init %i32 = call %specific_fn.loc11_31.2(%int_42.loc11_29) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc11_31.5: init %i32 = converted %int_42.loc11_29, %int.convert_checked.loc11_31.2 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_31.6: ref %i32 = array_index file.%c.var, %int_1
// CHECK:STDOUT:   %.loc11_31.7: init %i32 = initialize_from %.loc11_31.5 to %.loc11_31.6 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc11_31.8: init %array_type = array_init (%.loc11_31.4, %.loc11_31.7) to file.%c.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_1: init %array_type = converted %.loc11_31.1, %.loc11_31.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%c.var, %.loc11_1
// CHECK:STDOUT:   %c.ref: ref %array_type = name_ref c, file.%c
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %impl.elem0.loc16_16.1: %.45d = impl_witness_access constants.%impl_witness.0f6, element0 [concrete = constants.%Op.bba]
// CHECK:STDOUT:   %bound_method.loc16_16.1: <bound method> = bound_method %int_10, %impl.elem0.loc16_16.1 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %bound_method.loc16_16.1(%int_10) [concrete = constants.%int_-10.06d]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16_16.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_16.2: <bound method> = bound_method %int.snegate, %impl.elem0.loc16_16.2 [concrete = constants.%Convert.bound.1ca]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %bound_method.loc16_16.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.4be]
// CHECK:STDOUT:   %.loc16_16.1: Core.IntLiteral = value_of_initializer %int.snegate [concrete = constants.%int_-10.06d]
// CHECK:STDOUT:   %.loc16_16.2: Core.IntLiteral = converted %int.snegate, %.loc16_16.1 [concrete = constants.%int_-10.06d]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %specific_fn.loc16(%.loc16_16.2) [concrete = constants.%int_-10.c17]
// CHECK:STDOUT:   %.loc16_16.3: %i32 = value_of_initializer %int.convert_checked.loc16 [concrete = constants.%int_-10.c17]
// CHECK:STDOUT:   %.loc16_16.4: %i32 = converted %int.snegate, %.loc16_16.3 [concrete = constants.%int_-10.c17]
// CHECK:STDOUT:   %.loc16_19.1: ref %i32 = array_index %c.ref, %.loc16_16.4 [concrete = <error>]
// CHECK:STDOUT:   %.loc16_19.2: %i32 = bind_value %.loc16_19.1
// CHECK:STDOUT:   assign file.%d.var, %.loc16_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
