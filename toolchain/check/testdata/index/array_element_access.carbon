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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/array_element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/array_element_access.carbon

var a: array(i32, 2) = (12, 24);
var b: i32 = 1;
var c: i32 = a[0];
var d: i32 = a[b];

// CHECK:STDOUT: --- array_element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.ae7: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %int_24.e3c: Core.IntLiteral = int_value 24 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b4: <bound method> = bound_method %int_12.6a3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.827: <bound method> = bound_method %int_12.6a3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_12.1e1: %i32 = int_value 12 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.527: <bound method> = bound_method %int_24.e3c, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.3c4: <bound method> = bound_method %int_24.e3c, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_24.365: %i32 = int_value 24 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_12.1e1, %int_24.365) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.b59: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.698: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f0b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.af5: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.87e: %Int.as.Copy.impl.Op.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.26d: %Copy.type = facet_value %i32, (%Copy.impl_witness.f0b) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.26d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.87e, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.ae7 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.ae7 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc15: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, %i32.loc15 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc16: type = splice_block %i32.loc16 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7ce = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc17: type = splice_block %i32.loc17 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.7ce = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc18: type = splice_block %i32.loc18 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24.e3c]
// CHECK:STDOUT:   %.loc15_31.1: %tuple.type.f94 = tuple_literal (%int_12, %int_24)
// CHECK:STDOUT:   %impl.elem0.loc15_31.1: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc15_31.1: <bound method> = bound_method %int_12, %impl.elem0.loc15_31.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0b4]
// CHECK:STDOUT:   %specific_fn.loc15_31.1: <specific function> = specific_function %impl.elem0.loc15_31.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_31.2: <bound method> = bound_method %int_12, %specific_fn.loc15_31.1 [concrete = constants.%bound_method.827]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_31.1: init %i32 = call %bound_method.loc15_31.2(%int_12) [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc15_31.2: init %i32 = converted %int_12, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_31.1 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %int_0.loc15: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc15_31.3: ref %i32 = array_index file.%a.var, %int_0.loc15
// CHECK:STDOUT:   %.loc15_31.4: init %i32 = initialize_from %.loc15_31.2 to %.loc15_31.3 [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %impl.elem0.loc15_31.2: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc15_31.3: <bound method> = bound_method %int_24, %impl.elem0.loc15_31.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.527]
// CHECK:STDOUT:   %specific_fn.loc15_31.2: <specific function> = specific_function %impl.elem0.loc15_31.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_31.4: <bound method> = bound_method %int_24, %specific_fn.loc15_31.2 [concrete = constants.%bound_method.3c4]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_31.2: init %i32 = call %bound_method.loc15_31.4(%int_24) [concrete = constants.%int_24.365]
// CHECK:STDOUT:   %.loc15_31.5: init %i32 = converted %int_24, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_31.2 [concrete = constants.%int_24.365]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc15_31.6: ref %i32 = array_index file.%a.var, %int_1.loc15
// CHECK:STDOUT:   %.loc15_31.7: init %i32 = initialize_from %.loc15_31.5 to %.loc15_31.6 [concrete = constants.%int_24.365]
// CHECK:STDOUT:   %.loc15_31.8: init %array_type = array_init (%.loc15_31.4, %.loc15_31.7) to file.%a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc15_1: init %array_type = converted %.loc15_31.1, %.loc15_31.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc15_1
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc16: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc16_1.1: <bound method> = bound_method %int_1.loc16, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_1.2: <bound method> = bound_method %int_1.loc16, %specific_fn.loc16 [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i32 = call %bound_method.loc16_1.2(%int_1.loc16) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16: init %i32 = converted %int_1.loc16, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign file.%b.var, %.loc16
// CHECK:STDOUT:   %a.ref.loc17: ref %array_type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_0.loc17: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17_16: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc17_16.1: <bound method> = bound_method %int_0.loc17, %impl.elem0.loc17_16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.7cb]
// CHECK:STDOUT:   %specific_fn.loc17_16: <specific function> = specific_function %impl.elem0.loc17_16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_16.2: <bound method> = bound_method %int_0.loc17, %specific_fn.loc17_16 [concrete = constants.%bound_method.698]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17: init %i32 = call %bound_method.loc17_16.2(%int_0.loc17) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_16.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc17 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_16.2: %i32 = converted %int_0.loc17, %.loc17_16.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc17_17.1: ref %i32 = array_index %a.ref.loc17, %.loc17_16.2
// CHECK:STDOUT:   %.loc17_17.2: %i32 = bind_value %.loc17_17.1
// CHECK:STDOUT:   %impl.elem0.loc17_17: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc17_17.1: <bound method> = bound_method %.loc17_17.2, %impl.elem0.loc17_17
// CHECK:STDOUT:   %specific_fn.loc17_17: <specific function> = specific_function %impl.elem0.loc17_17, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_17.2: <bound method> = bound_method %.loc17_17.2, %specific_fn.loc17_17
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc17: init %i32 = call %bound_method.loc17_17.2(%.loc17_17.2)
// CHECK:STDOUT:   assign file.%c.var, %Int.as.Copy.impl.Op.call.loc17
// CHECK:STDOUT:   %a.ref.loc18: ref %array_type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %b.ref: ref %i32 = name_ref b, file.%b [concrete = file.%b.var]
// CHECK:STDOUT:   %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc18_16: %i32 = bind_value %b.ref
// CHECK:STDOUT:   %.loc18_17.1: ref %i32 = array_index %a.ref.loc18, %.loc18_16
// CHECK:STDOUT:   %.loc18_17.2: %i32 = bind_value %.loc18_17.1
// CHECK:STDOUT:   %impl.elem0.loc18: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc18_17.1: <bound method> = bound_method %.loc18_17.2, %impl.elem0.loc18
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_17.2: <bound method> = bound_method %.loc18_17.2, %specific_fn.loc18
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc18: init %i32 = call %bound_method.loc18_17.2(%.loc18_17.2)
// CHECK:STDOUT:   assign file.%d.var, %Int.as.Copy.impl.Op.call.loc18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
