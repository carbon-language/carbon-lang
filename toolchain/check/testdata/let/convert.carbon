// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/convert.carbon

fn F() -> i32 {
  var v: (i32, i32, i32) = (1, 2, 3);
  // Convert from object representation to value representation.
  let w: (i32, i32, i32) = v;
  return w.1;
}

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.dc6: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.0db: <bound method> = bound_method %int_3.1ba, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.456: <specific function> = specific_function %Convert.bound.0db, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.25b: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.dc6 = tuple_value (%int_1.c60, %int_2.166, %int_3.25b) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %tuple.type.dc6 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.dc6 = bind_name v, %v.var
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc12_36.1: %tuple.type.37f = tuple_literal (%int_1.loc12, %int_2, %int_3)
// CHECK:STDOUT:   %impl.elem0.loc12_36.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12_36.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_36.1 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_36.1: <specific function> = specific_function %Convert.bound.loc12_36.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.1: init %i32 = call %Convert.specific_fn.loc12_36.1(%int_1.loc12) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc12_36.2: init %i32 = converted %int_1.loc12, %int.convert_checked.loc12_36.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %tuple.elem0.loc12: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc12_36.3: init %i32 = initialize_from %.loc12_36.2 to %tuple.elem0.loc12 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc12_36.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12_36.2: <bound method> = bound_method %int_2, %impl.elem0.loc12_36.2 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_36.2: <specific function> = specific_function %Convert.bound.loc12_36.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.2: init %i32 = call %Convert.specific_fn.loc12_36.2(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc12_36.4: init %i32 = converted %int_2, %int.convert_checked.loc12_36.2 [template = constants.%int_2.166]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %v.var, element1
// CHECK:STDOUT:   %.loc12_36.5: init %i32 = initialize_from %.loc12_36.4 to %tuple.elem1.loc12 [template = constants.%int_2.166]
// CHECK:STDOUT:   %impl.elem0.loc12_36.3: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc12_36.3: <bound method> = bound_method %int_3, %impl.elem0.loc12_36.3 [template = constants.%Convert.bound.0db]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_36.3: <specific function> = specific_function %Convert.bound.loc12_36.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.456]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.3: init %i32 = call %Convert.specific_fn.loc12_36.3(%int_3) [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc12_36.6: init %i32 = converted %int_3, %int.convert_checked.loc12_36.3 [template = constants.%int_3.25b]
// CHECK:STDOUT:   %tuple.elem2.loc12: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc12_36.7: init %i32 = initialize_from %.loc12_36.6 to %tuple.elem2.loc12 [template = constants.%int_3.25b]
// CHECK:STDOUT:   %.loc12_36.8: init %tuple.type.dc6 = tuple_init (%.loc12_36.3, %.loc12_36.5, %.loc12_36.7) to %v.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_37: init %tuple.type.dc6 = converted %.loc12_36.1, %.loc12_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %v.var, %.loc12_37
// CHECK:STDOUT:   %v.ref: ref %tuple.type.dc6 = name_ref v, %v
// CHECK:STDOUT:   %tuple.elem0.loc14: ref %i32 = tuple_access %v.ref, element0
// CHECK:STDOUT:   %.loc14_28.1: %i32 = bind_value %tuple.elem0.loc14
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %v.ref, element1
// CHECK:STDOUT:   %.loc14_28.2: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   %tuple.elem2.loc14: ref %i32 = tuple_access %v.ref, element2
// CHECK:STDOUT:   %.loc14_28.3: %i32 = bind_value %tuple.elem2.loc14
// CHECK:STDOUT:   %tuple: %tuple.type.dc6 = tuple_value (%.loc14_28.1, %.loc14_28.2, %.loc14_28.3)
// CHECK:STDOUT:   %.loc14_29: %tuple.type.dc6 = converted %v.ref, %tuple
// CHECK:STDOUT:   %w: %tuple.type.dc6 = bind_name w, %.loc14_29
// CHECK:STDOUT:   %w.ref: %tuple.type.dc6 = name_ref w, %w
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc15: %i32 = tuple_access %w.ref, element1
// CHECK:STDOUT:   return %tuple.elem1.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
