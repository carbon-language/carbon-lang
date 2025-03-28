// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/element_access.carbon

var a: (i32,) = (12,);
var b: (i32,) = a;
var c: i32 = b.0;

// CHECK:STDOUT: --- element_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.85c: type = tuple_type (type) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %int_12.6a3: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_12.6a3, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_12.6a3, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_12.1e1: %i32 = int_value 12 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.a1c = tuple_value (%int_12.1e1) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %tuple.type.a1c = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %tuple.type.a1c = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %tuple.type.a1c = var a
// CHECK:STDOUT:   %.loc11_13.1: type = splice_block %.loc11_13.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_13.2: %tuple.type.85c = tuple_literal (%i32.loc11)
// CHECK:STDOUT:     %.loc11_13.3: type = converted %.loc11_13.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.a1c = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %tuple.type.a1c = binding_pattern b
// CHECK:STDOUT:     %.loc12_1: %tuple.type.a1c = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %tuple.type.a1c = var b
// CHECK:STDOUT:   %.loc12_13.1: type = splice_block %.loc12_13.3 [concrete = constants.%tuple.type.a1c] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_13.2: %tuple.type.85c = tuple_literal (%i32.loc12)
// CHECK:STDOUT:     %.loc12_13.3: type = converted %.loc12_13.2, constants.%tuple.type.a1c [concrete = constants.%tuple.type.a1c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %tuple.type.a1c = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %i32 = binding_pattern c
// CHECK:STDOUT:     %.loc13_1: %i32 = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var c
// CHECK:STDOUT:   %.loc13_8: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12.6a3]
// CHECK:STDOUT:   %.loc11_21.1: %tuple.type.985 = tuple_literal (%int_12)
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_21.1: <bound method> = bound_method %int_12, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_21.2: <bound method> = bound_method %int_12, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc11_21.2(%int_12) [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc11_21.2: init %i32 = converted %int_12, %int.convert_checked [concrete = constants.%int_12.1e1]
// CHECK:STDOUT:   %.loc11_21.3: init %tuple.type.a1c = tuple_init (%.loc11_21.2) to file.%a.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc11_1: init %tuple.type.a1c = converted %.loc11_21.1, %.loc11_21.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   %a.ref: ref %tuple.type.a1c = name_ref a, file.%a
// CHECK:STDOUT:   %tuple.elem0.loc12: ref %i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc12_17.1: %i32 = bind_value %tuple.elem0.loc12
// CHECK:STDOUT:   %.loc12_17.2: init %tuple.type.a1c = tuple_init (%.loc12_17.1) to file.%b.var
// CHECK:STDOUT:   %.loc12_1: init %tuple.type.a1c = converted %a.ref, %.loc12_17.2
// CHECK:STDOUT:   assign file.%b.var, %.loc12_1
// CHECK:STDOUT:   %b.ref: ref %tuple.type.a1c = name_ref b, file.%b
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref %i32 = tuple_access %b.ref, element0
// CHECK:STDOUT:   %.loc13: %i32 = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
