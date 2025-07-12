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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/fail_negative_indexing.carbon

var c: array(i32, 2) = (42, 42);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+4]]:16: error: array index `-10` is past the end of type `array(i32, 2)` [ArrayIndexOutOfBounds]
// CHECK:STDERR: var d: i32 = c[-10];
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
var d: i32 = c[-10];


// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.132: <bound method> = bound_method %int_42.20e, %Convert.956 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.8c1: <bound method> = bound_method %int_42.20e, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_42.c68, %int_42.c68) [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Negate.impl_witness.973: <witness> = impl_witness imports.%Negate.impl_witness_table.b22 [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness.973) [concrete]
// CHECK:STDOUT:   %.d3d: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.type.18a: type = fn_type @Op.3 [concrete]
// CHECK:STDOUT:   %Op.99f: %Op.type.18a = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_10, %Op.99f [concrete]
// CHECK:STDOUT:   %int_-10.06d: Core.IntLiteral = int_value -10 [concrete]
// CHECK:STDOUT:   %Convert.bound.1ca: <bound method> = bound_method %int_-10.06d, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.af0: <bound method> = bound_method %int_-10.06d, %Convert.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc13_50, unloaded
// CHECK:STDOUT:   %Core.import_ref.5e6: %Op.type.18a = import_ref Core//prelude/parts/int_literal, loc14_31, loaded [concrete = constants.%Op.99f]
// CHECK:STDOUT:   %Negate.impl_witness_table.b22 = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.5e6), @Negate.impl.0ef [concrete]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.ae7 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.ae7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %array_type = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %i32.loc15 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %array_type = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.7ce = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc20: type = splice_block %i32.loc20 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_42.loc15_25: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %int_42.loc15_29: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %.loc15_31.1: %tuple.type = tuple_literal (%int_42.loc15_25, %int_42.loc15_29)
// CHECK:STDOUT:   %impl.elem0.loc15_31.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_31.1: <bound method> = bound_method %int_42.loc15_25, %impl.elem0.loc15_31.1 [concrete = constants.%Convert.bound.132]
// CHECK:STDOUT:   %specific_fn.loc15_31.1: <specific function> = specific_function %impl.elem0.loc15_31.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_31.2: <bound method> = bound_method %int_42.loc15_25, %specific_fn.loc15_31.1 [concrete = constants.%bound_method.8c1]
// CHECK:STDOUT:   %int.convert_checked.loc15_31.1: init %i32 = call %bound_method.loc15_31.2(%int_42.loc15_25) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc15_31.2: init %i32 = converted %int_42.loc15_25, %int.convert_checked.loc15_31.1 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc15_31.3: ref %i32 = array_index file.%c.var, %int_0
// CHECK:STDOUT:   %.loc15_31.4: init %i32 = initialize_from %.loc15_31.2 to %.loc15_31.3 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %impl.elem0.loc15_31.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_31.3: <bound method> = bound_method %int_42.loc15_29, %impl.elem0.loc15_31.2 [concrete = constants.%Convert.bound.132]
// CHECK:STDOUT:   %specific_fn.loc15_31.2: <specific function> = specific_function %impl.elem0.loc15_31.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_31.4: <bound method> = bound_method %int_42.loc15_29, %specific_fn.loc15_31.2 [concrete = constants.%bound_method.8c1]
// CHECK:STDOUT:   %int.convert_checked.loc15_31.2: init %i32 = call %bound_method.loc15_31.4(%int_42.loc15_29) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc15_31.5: init %i32 = converted %int_42.loc15_29, %int.convert_checked.loc15_31.2 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc15_31.6: ref %i32 = array_index file.%c.var, %int_1
// CHECK:STDOUT:   %.loc15_31.7: init %i32 = initialize_from %.loc15_31.5 to %.loc15_31.6 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc15_31.8: init %array_type = array_init (%.loc15_31.4, %.loc15_31.7) to file.%c.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc15_1: init %array_type = converted %.loc15_31.1, %.loc15_31.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%c.var, %.loc15_1
// CHECK:STDOUT:   %c.ref: ref %array_type = name_ref c, file.%c [concrete = file.%c.var]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %impl.elem1: %.d3d = impl_witness_access constants.%Negate.impl_witness.973, element1 [concrete = constants.%Op.99f]
// CHECK:STDOUT:   %bound_method.loc20_16.1: <bound method> = bound_method %int_10, %impl.elem1 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %bound_method.loc20_16.1(%int_10) [concrete = constants.%int_-10.06d]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc20: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc20_16.2: <bound method> = bound_method %int.snegate, %impl.elem0.loc20 [concrete = constants.%Convert.bound.1ca]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_16.3: <bound method> = bound_method %int.snegate, %specific_fn.loc20 [concrete = constants.%bound_method.af0]
// CHECK:STDOUT:   %.loc20_16.1: Core.IntLiteral = value_of_initializer %int.snegate [concrete = constants.%int_-10.06d]
// CHECK:STDOUT:   %.loc20_16.2: Core.IntLiteral = converted %int.snegate, %.loc20_16.1 [concrete = constants.%int_-10.06d]
// CHECK:STDOUT:   %int.convert_checked.loc20: init %i32 = call %bound_method.loc20_16.3(%.loc20_16.2) [concrete = constants.%int_-10.c17]
// CHECK:STDOUT:   %.loc20_16.3: %i32 = value_of_initializer %int.convert_checked.loc20 [concrete = constants.%int_-10.c17]
// CHECK:STDOUT:   %.loc20_16.4: %i32 = converted %int.snegate, %.loc20_16.3 [concrete = constants.%int_-10.c17]
// CHECK:STDOUT:   %.loc20_19.1: ref %i32 = array_index %c.ref, %.loc20_16.4 [concrete = <error>]
// CHECK:STDOUT:   %.loc20_19.2: %i32 = bind_value %.loc20_19.1 [concrete = <error>]
// CHECK:STDOUT:   assign file.%d.var, %.loc20_19.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
