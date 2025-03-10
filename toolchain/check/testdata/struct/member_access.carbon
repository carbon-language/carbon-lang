// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/member_access.carbon

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [concrete]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.ddf: type = struct_type {.a: f64, .b: %i32} [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.fbb: type = struct_type {.a: f64, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.b.ddf = struct_value (%float, %int_1.5d2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:     .z = %z
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %struct_type.a.b.ddf = binding_pattern x
// CHECK:STDOUT:     %.loc11_1: %struct_type.a.b.ddf = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.a.b.ddf = var x
// CHECK:STDOUT:   %.loc11_25: type = splice_block %struct_type.a.b [concrete = constants.%struct_type.a.b.ddf] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %float.make_type: init type = call constants.%Float(%int_64) [concrete = f64]
// CHECK:STDOUT:     %.loc11_13.1: type = value_of_initializer %float.make_type [concrete = f64]
// CHECK:STDOUT:     %.loc11_13.2: type = converted %float.make_type, %.loc11_13.1 [concrete = f64]
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b: type = struct_type {.a: f64, .b: %i32} [concrete = constants.%struct_type.a.b.ddf]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.a.b.ddf = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %i32 = binding_pattern y
// CHECK:STDOUT:     %.loc12_1: %i32 = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %i32 = var y
// CHECK:STDOUT:   %.loc12_8: type = splice_block %i32.loc12 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %i32 = bind_name y, %y.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %i32 = binding_pattern z
// CHECK:STDOUT:     %.loc13_1: %i32 = var_pattern %z.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z.var: ref %i32 = var z
// CHECK:STDOUT:   %.loc13_8: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %z: ref %i32 = bind_name z, %z.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %float: f64 = float_literal 0 [concrete = constants.%float]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_46.1: %struct_type.a.b.fbb = struct_literal (%float, %int_1)
// CHECK:STDOUT:   %.loc11_46.2: ref f64 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_46.3: init f64 = initialize_from %float to %.loc11_46.2 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_46.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_46.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc11_46.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_46.4: init %i32 = converted %int_1, %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_46.5: ref %i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_46.6: init %i32 = initialize_from %.loc11_46.4 to %.loc11_46.5 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_46.7: init %struct_type.a.b.ddf = struct_init (%.loc11_46.3, %.loc11_46.6) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc11_1: init %struct_type.a.b.ddf = converted %.loc11_46.1, %.loc11_46.7 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.a.b.ddf = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_15.1: ref %i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc12_15.2: %i32 = bind_value %.loc12_15.1
// CHECK:STDOUT:   assign file.%y.var, %.loc12_15.2
// CHECK:STDOUT:   %y.ref: ref %i32 = name_ref y, file.%y
// CHECK:STDOUT:   %.loc13: %i32 = bind_value %y.ref
// CHECK:STDOUT:   assign file.%z.var, %.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
