// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/array_vs_tuple.carbon

fn G() {
  // These should have two different constant values.
  var a: [i32; 3] = (1, 2, 3);
  var b: (i32, i32, i32) = (1, 2, 3);
}

// CHECK:STDOUT: --- array_vs_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_3.1baa: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1baa, %i32 [template]
// CHECK:STDOUT:   %int_1.5b81: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc8: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.37fc: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.cd1a: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b05: <witness> = impl_witness (imports.%import_ref.7238), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.4666: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.9251: %Convert.type.4666 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd1: <bound method> = bound_method %int_1.5b81, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b730: <specific function> = specific_function %Convert.bound.afd1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c605: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.f0b0: <bound method> = bound_method %int_2.ecc8, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e1: <specific function> = specific_function %Convert.bound.f0b0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.1667: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.0db0: <bound method> = bound_method %int_3.1baa, %Convert.9251 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4567: <specific function> = specific_function %Convert.bound.0db0, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25ba: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.c605, %int_2.1667, %int_3.25ba) [template]
// CHECK:STDOUT:   %tuple.type.dc60: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.dc60 = tuple_value (%int_1.c605, %int_2.1667, %int_3.25ba) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a691
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT:   %int_1.loc13_22: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %int_2.loc13_25: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %int_3.loc13: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %.loc13_29.1: %tuple.type.37fc = tuple_literal (%int_1.loc13_22, %int_2.loc13_25, %int_3.loc13)
// CHECK:STDOUT:   %impl.elem0.loc13_29.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc13_29.1: <bound method> = bound_method %int_1.loc13_22, %impl.elem0.loc13_29.1 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_29.1: <specific function> = specific_function %Convert.bound.loc13_29.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.1: init %i32 = call %Convert.specific_fn.loc13_29.1(%int_1.loc13_22) [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc13_29.2: init %i32 = converted %int_1.loc13_22, %int.convert_checked.loc13_29.1 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc13_29.3: ref %i32 = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc13_29.4: init %i32 = initialize_from %.loc13_29.2 to %.loc13_29.3 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %impl.elem0.loc13_29.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc13_29.2: <bound method> = bound_method %int_2.loc13_25, %impl.elem0.loc13_29.2 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_29.2: <specific function> = specific_function %Convert.bound.loc13_29.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.2: init %i32 = call %Convert.specific_fn.loc13_29.2(%int_2.loc13_25) [template = constants.%int_2.1667]
// CHECK:STDOUT:   %.loc13_29.5: init %i32 = converted %int_2.loc13_25, %int.convert_checked.loc13_29.2 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %int_1.loc13_29: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %.loc13_29.6: ref %i32 = array_index %a.var, %int_1.loc13_29
// CHECK:STDOUT:   %.loc13_29.7: init %i32 = initialize_from %.loc13_29.5 to %.loc13_29.6 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %impl.elem0.loc13_29.3: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc13_29.3: <bound method> = bound_method %int_3.loc13, %impl.elem0.loc13_29.3 [template = constants.%Convert.bound.0db0]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_29.3: <specific function> = specific_function %Convert.bound.loc13_29.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4567]
// CHECK:STDOUT:   %int.convert_checked.loc13_29.3: init %i32 = call %Convert.specific_fn.loc13_29.3(%int_3.loc13) [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc13_29.8: init %i32 = converted %int_3.loc13, %int.convert_checked.loc13_29.3 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %int_2.loc13_29: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %.loc13_29.9: ref %i32 = array_index %a.var, %int_2.loc13_29
// CHECK:STDOUT:   %.loc13_29.10: init %i32 = initialize_from %.loc13_29.8 to %.loc13_29.9 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc13_29.11: init %array_type = array_init (%.loc13_29.4, %.loc13_29.7, %.loc13_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc13_30: init %array_type = converted %.loc13_29.1, %.loc13_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc13_30
// CHECK:STDOUT:   %b.var: ref %tuple.type.dc60 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.dc60 = bind_name b, %b.var
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b81]
// CHECK:STDOUT:   %int_2.loc14: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc8]
// CHECK:STDOUT:   %int_3.loc14: Core.IntLiteral = int_value 3 [template = constants.%int_3.1baa]
// CHECK:STDOUT:   %.loc14_36.1: %tuple.type.37fc = tuple_literal (%int_1.loc14, %int_2.loc14, %int_3.loc14)
// CHECK:STDOUT:   %impl.elem0.loc14_36.1: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc14_36.1: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_36.1 [template = constants.%Convert.bound.afd1]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_36.1: <specific function> = specific_function %Convert.bound.loc14_36.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b730]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.1: init %i32 = call %Convert.specific_fn.loc14_36.1(%int_1.loc14) [template = constants.%int_1.c605]
// CHECK:STDOUT:   %.loc14_36.2: init %i32 = converted %int_1.loc14, %int.convert_checked.loc14_36.1 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc14_36.3: init %i32 = initialize_from %.loc14_36.2 to %tuple.elem0 [template = constants.%int_1.c605]
// CHECK:STDOUT:   %impl.elem0.loc14_36.2: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc14_36.2: <bound method> = bound_method %int_2.loc14, %impl.elem0.loc14_36.2 [template = constants.%Convert.bound.f0b0]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_36.2: <specific function> = specific_function %Convert.bound.loc14_36.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e1]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.2: init %i32 = call %Convert.specific_fn.loc14_36.2(%int_2.loc14) [template = constants.%int_2.1667]
// CHECK:STDOUT:   %.loc14_36.4: init %i32 = converted %int_2.loc14, %int.convert_checked.loc14_36.2 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc14_36.5: init %i32 = initialize_from %.loc14_36.4 to %tuple.elem1 [template = constants.%int_2.1667]
// CHECK:STDOUT:   %impl.elem0.loc14_36.3: %Convert.type.cd1a = impl_witness_access constants.%impl_witness.5b05, element0 [template = constants.%Convert.9251]
// CHECK:STDOUT:   %Convert.bound.loc14_36.3: <bound method> = bound_method %int_3.loc14, %impl.elem0.loc14_36.3 [template = constants.%Convert.bound.0db0]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_36.3: <specific function> = specific_function %Convert.bound.loc14_36.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4567]
// CHECK:STDOUT:   %int.convert_checked.loc14_36.3: init %i32 = call %Convert.specific_fn.loc14_36.3(%int_3.loc14) [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc14_36.6: init %i32 = converted %int_3.loc14, %int.convert_checked.loc14_36.3 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %b.var, element2
// CHECK:STDOUT:   %.loc14_36.7: init %i32 = initialize_from %.loc14_36.6 to %tuple.elem2 [template = constants.%int_3.25ba]
// CHECK:STDOUT:   %.loc14_36.8: init %tuple.type.dc60 = tuple_init (%.loc14_36.3, %.loc14_36.5, %.loc14_36.7) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_37: init %tuple.type.dc60 = converted %.loc14_36.1, %.loc14_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc14_37
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
