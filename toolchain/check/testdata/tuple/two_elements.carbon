// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/two_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/two_elements.carbon

let v: (i32, i32) = (4, 102);
let w: (i32, i32) = v;

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_102.b54: Core.IntLiteral = int_value 102 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.063: <bound method> = bound_method %int_102.b54, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.c20: <bound method> = bound_method %int_102.b54, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_102.b91: %i32 = int_value 102 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%int_4.940, %int_102.b91) [concrete]
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
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:     .w = %w
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %tuple.type.d07 = binding_pattern v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_17.1: type = splice_block %.loc11_17.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc11_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_17.2: %tuple.type.24b = tuple_literal (%i32.loc11_9, %i32.loc11_14)
// CHECK:STDOUT:     %.loc11_17.3: type = converted %.loc11_17.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11_28.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_28.1: <bound method> = bound_method @__global_init.%int_4.loc11, %impl.elem0.loc11_28.1 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc11_28.1: <specific function> = specific_function %impl.elem0.loc11_28.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_28.2: <bound method> = bound_method @__global_init.%int_4.loc11, %specific_fn.loc11_28.1 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.1: init %i32 = call %bound_method.loc11_28.2(@__global_init.%int_4.loc11) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_28.1: %i32 = value_of_initializer %int.convert_checked.loc11_28.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_28.2: %i32 = converted @__global_init.%int_4.loc11, %.loc11_28.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %impl.elem0.loc11_28.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_28.3: <bound method> = bound_method @__global_init.%int_102.loc11, %impl.elem0.loc11_28.2 [concrete = constants.%Convert.bound.063]
// CHECK:STDOUT:   %specific_fn.loc11_28.2: <specific function> = specific_function %impl.elem0.loc11_28.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_28.4: <bound method> = bound_method @__global_init.%int_102.loc11, %specific_fn.loc11_28.2 [concrete = constants.%bound_method.c20]
// CHECK:STDOUT:   %int.convert_checked.loc11_28.2: init %i32 = call %bound_method.loc11_28.4(@__global_init.%int_102.loc11) [concrete = constants.%int_102.b91]
// CHECK:STDOUT:   %.loc11_28.3: %i32 = value_of_initializer %int.convert_checked.loc11_28.2 [concrete = constants.%int_102.b91]
// CHECK:STDOUT:   %.loc11_28.4: %i32 = converted @__global_init.%int_102.loc11, %.loc11_28.3 [concrete = constants.%int_102.b91]
// CHECK:STDOUT:   %tuple: %tuple.type.d07 = tuple_value (%.loc11_28.2, %.loc11_28.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_28.5: %tuple.type.d07 = converted @__global_init.%.loc11, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %v: %tuple.type.d07 = bind_name v, %.loc11_28.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %tuple.type.d07 = binding_pattern w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_17.1: type = splice_block %.loc12_17.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc12_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_17.2: %tuple.type.24b = tuple_literal (%i32.loc12_9, %i32.loc12_14)
// CHECK:STDOUT:     %.loc12_17.3: type = converted %.loc12_17.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: %tuple.type.d07 = bind_name w, @__global_init.%v.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %tuple.type.d07 = binding_pattern x
// CHECK:STDOUT:     %.loc14_1: %tuple.type.d07 = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type.d07 = var x
// CHECK:STDOUT:   %.loc14_17.1: type = splice_block %.loc14_17.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc14_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_17.2: %tuple.type.24b = tuple_literal (%i32.loc14_9, %i32.loc14_14)
// CHECK:STDOUT:     %.loc14_17.3: type = converted %.loc14_17.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.d07 = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %tuple.type.d07 = binding_pattern y
// CHECK:STDOUT:     %.loc15_1: %tuple.type.d07 = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %tuple.type.d07 = var y
// CHECK:STDOUT:   %.loc15_17.1: type = splice_block %.loc15_17.3 [concrete = constants.%tuple.type.d07] {
// CHECK:STDOUT:     %int_32.loc15_9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc15_17.2: %tuple.type.24b = tuple_literal (%i32.loc15_9, %i32.loc15_14)
// CHECK:STDOUT:     %.loc15_17.3: type = converted %.loc15_17.2, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %tuple.type.d07 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4.loc11: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_102.loc11: Core.IntLiteral = int_value 102 [concrete = constants.%int_102.b54]
// CHECK:STDOUT:   %.loc11: %tuple.type.f94 = tuple_literal (%int_4.loc11, %int_102.loc11)
// CHECK:STDOUT:   %v.ref: %tuple.type.d07 = name_ref v, file.%v
// CHECK:STDOUT:   %int_4.loc14: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_102.loc14: Core.IntLiteral = int_value 102 [concrete = constants.%int_102.b54]
// CHECK:STDOUT:   %.loc14_28.1: %tuple.type.f94 = tuple_literal (%int_4.loc14, %int_102.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_28.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_28.1: <bound method> = bound_method %int_4.loc14, %impl.elem0.loc14_28.1 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc14_28.1: <specific function> = specific_function %impl.elem0.loc14_28.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_28.2: <bound method> = bound_method %int_4.loc14, %specific_fn.loc14_28.1 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.1: init %i32 = call %bound_method.loc14_28.2(%int_4.loc14) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc14_28.2: init %i32 = converted %int_4.loc14, %int.convert_checked.loc14_28.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %tuple.elem0.loc14: ref %i32 = tuple_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_28.3: init %i32 = initialize_from %.loc14_28.2 to %tuple.elem0.loc14 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %impl.elem0.loc14_28.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_28.3: <bound method> = bound_method %int_102.loc14, %impl.elem0.loc14_28.2 [concrete = constants.%Convert.bound.063]
// CHECK:STDOUT:   %specific_fn.loc14_28.2: <specific function> = specific_function %impl.elem0.loc14_28.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_28.4: <bound method> = bound_method %int_102.loc14, %specific_fn.loc14_28.2 [concrete = constants.%bound_method.c20]
// CHECK:STDOUT:   %int.convert_checked.loc14_28.2: init %i32 = call %bound_method.loc14_28.4(%int_102.loc14) [concrete = constants.%int_102.b91]
// CHECK:STDOUT:   %.loc14_28.4: init %i32 = converted %int_102.loc14, %int.convert_checked.loc14_28.2 [concrete = constants.%int_102.b91]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_28.5: init %i32 = initialize_from %.loc14_28.4 to %tuple.elem1.loc14 [concrete = constants.%int_102.b91]
// CHECK:STDOUT:   %.loc14_28.6: init %tuple.type.d07 = tuple_init (%.loc14_28.3, %.loc14_28.5) to file.%x.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc14_1: init %tuple.type.d07 = converted %.loc14_28.1, %.loc14_28.6 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_1
// CHECK:STDOUT:   %x.ref: ref %tuple.type.d07 = name_ref x, file.%x
// CHECK:STDOUT:   %tuple.elem0.loc15_21.1: ref %i32 = tuple_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_21.1: %i32 = bind_value %tuple.elem0.loc15_21.1
// CHECK:STDOUT:   %tuple.elem0.loc15_21.2: ref %i32 = tuple_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_21.2: init %i32 = initialize_from %.loc15_21.1 to %tuple.elem0.loc15_21.2
// CHECK:STDOUT:   %tuple.elem1.loc15_21.1: ref %i32 = tuple_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_21.3: %i32 = bind_value %tuple.elem1.loc15_21.1
// CHECK:STDOUT:   %tuple.elem1.loc15_21.2: ref %i32 = tuple_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_21.4: init %i32 = initialize_from %.loc15_21.3 to %tuple.elem1.loc15_21.2
// CHECK:STDOUT:   %.loc15_21.5: init %tuple.type.d07 = tuple_init (%.loc15_21.2, %.loc15_21.4) to file.%y.var
// CHECK:STDOUT:   %.loc15_1: init %tuple.type.d07 = converted %x.ref, %.loc15_21.5
// CHECK:STDOUT:   assign file.%y.var, %.loc15_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
