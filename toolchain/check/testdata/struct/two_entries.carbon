// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/two_entries.carbon

let v: {.a: i32, .b: i32} = {.a = 1, .b = 2};
let w: {.a: i32, .b: i32} = v;

var x: {.a: i32, .b: i32} = {.a = 1, .b = 2};
var y: {.a: i32, .b: i32} = x;

// CHECK:STDOUT: --- two_entries.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.501: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.cfd: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%int_1.5d2, %int_2.ef8) [concrete]
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
// CHECK:STDOUT:     %v.patt: %struct_type.a.b.501 = binding_pattern v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_25: type = splice_block %struct_type.a.b.loc11 [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc11_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc11: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl.elem0.loc11_44.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_44.1: <bound method> = bound_method @__global_init.%int_1.loc11, %impl.elem0.loc11_44.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc11_44.1: <specific function> = specific_function %impl.elem0.loc11_44.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_44.2: <bound method> = bound_method @__global_init.%int_1.loc11, %specific_fn.loc11_44.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.1: init %i32 = call %bound_method.loc11_44.2(@__global_init.%int_1.loc11) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_44.1: %i32 = value_of_initializer %int.convert_checked.loc11_44.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_44.2: %i32 = converted @__global_init.%int_1.loc11, %.loc11_44.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_44.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_44.3: <bound method> = bound_method @__global_init.%int_2.loc11, %impl.elem0.loc11_44.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc11_44.2: <specific function> = specific_function %impl.elem0.loc11_44.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_44.4: <bound method> = bound_method @__global_init.%int_2.loc11, %specific_fn.loc11_44.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc11_44.2: init %i32 = call %bound_method.loc11_44.4(@__global_init.%int_2.loc11) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_44.3: %i32 = value_of_initializer %int.convert_checked.loc11_44.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_44.4: %i32 = converted @__global_init.%int_2.loc11, %.loc11_44.3 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %struct: %struct_type.a.b.501 = struct_value (%.loc11_44.2, %.loc11_44.4) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc11_44.5: %struct_type.a.b.501 = converted @__global_init.%.loc11, %struct [concrete = constants.%struct]
// CHECK:STDOUT:   %v: %struct_type.a.b.501 = bind_name v, %.loc11_44.5
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %struct_type.a.b.501 = binding_pattern w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: type = splice_block %struct_type.a.b.loc12 [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc12_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc12: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: %struct_type.a.b.501 = bind_name w, @__global_init.%v.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %struct_type.a.b.501 = binding_pattern x
// CHECK:STDOUT:     %.loc14_1: %struct_type.a.b.501 = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.501 = var x
// CHECK:STDOUT:   %.loc14_25: type = splice_block %struct_type.a.b.loc14 [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc14_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc14: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.501 = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %struct_type.a.b.501 = binding_pattern y
// CHECK:STDOUT:     %.loc15_1: %struct_type.a.b.501 = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.a.b.501 = var y
// CHECK:STDOUT:   %.loc15_25: type = splice_block %struct_type.a.b.loc15 [concrete = constants.%struct_type.a.b.501] {
// CHECK:STDOUT:     %int_32.loc15_13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc15_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.loc15: type = struct_type {.a: %i32, .b: %i32} [concrete = constants.%struct_type.a.b.501]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.a.b.501 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11: %struct_type.a.b.cfd = struct_literal (%int_1.loc11, %int_2.loc11)
// CHECK:STDOUT:   %v.ref: %struct_type.a.b.501 = name_ref v, file.%v
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc14_44.1: %struct_type.a.b.cfd = struct_literal (%int_1.loc14, %int_2.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_44.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_44.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_44.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc14_44.1: <specific function> = specific_function %impl.elem0.loc14_44.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_44.2: <bound method> = bound_method %int_1.loc14, %specific_fn.loc14_44.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.1: init %i32 = call %bound_method.loc14_44.2(%int_1.loc14) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_44.2: init %i32 = converted %int_1.loc14, %int.convert_checked.loc14_44.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc14_44.3: ref %i32 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc14_44.4: init %i32 = initialize_from %.loc14_44.2 to %.loc14_44.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc14_44.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_44.3: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_44.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc14_44.2: <specific function> = specific_function %impl.elem0.loc14_44.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_44.4: <bound method> = bound_method %int_2.loc14, %specific_fn.loc14_44.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc14_44.2: init %i32 = call %bound_method.loc14_44.4(%int_2.loc14) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_44.5: init %i32 = converted %int_2.loc14, %int.convert_checked.loc14_44.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_44.6: ref %i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc14_44.7: init %i32 = initialize_from %.loc14_44.5 to %.loc14_44.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc14_44.8: init %struct_type.a.b.501 = struct_init (%.loc14_44.4, %.loc14_44.7) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc14_1: init %struct_type.a.b.501 = converted %.loc14_44.1, %.loc14_44.8 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.501 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc15_29.1: ref %i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc15_29.2: %i32 = bind_value %.loc15_29.1
// CHECK:STDOUT:   %.loc15_29.3: ref %i32 = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc15_29.4: init %i32 = initialize_from %.loc15_29.2 to %.loc15_29.3
// CHECK:STDOUT:   %.loc15_29.5: ref %i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc15_29.6: %i32 = bind_value %.loc15_29.5
// CHECK:STDOUT:   %.loc15_29.7: ref %i32 = struct_access file.%y.var, element1
// CHECK:STDOUT:   %.loc15_29.8: init %i32 = initialize_from %.loc15_29.6 to %.loc15_29.7
// CHECK:STDOUT:   %.loc15_29.9: init %struct_type.a.b.501 = struct_init (%.loc15_29.4, %.loc15_29.8) to file.%y.var
// CHECK:STDOUT:   %.loc15_1: init %struct_type.a.b.501 = converted %x.ref, %.loc15_29.9
// CHECK:STDOUT:   assign file.%y.var, %.loc15_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
