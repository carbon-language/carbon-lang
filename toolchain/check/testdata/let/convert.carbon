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
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %interface.5: <witness> = interface_witness (%Convert.10) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_1.2, %int_2.2, %int_3.2) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.5
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
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc12_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc12_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12_16: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc12_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12_21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.1: %tuple.type.1 = tuple_literal (%i32.loc12_11, %i32.loc12_16, %i32.loc12_21)
// CHECK:STDOUT:   %.loc12_24.2: type = converted %.loc12_24.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %v.var: ref %tuple.type.2 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.2 = bind_name v, %v.var
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %.loc12_36.1: %tuple.type.3 = tuple_literal (%int_1.loc12, %int_2, %int_3)
// CHECK:STDOUT:   %impl.elem0.loc12_36.1: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_36.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_36.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_36.1: <specific function> = specific_function %Convert.bound.loc12_36.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.1: init %i32 = call %Convert.specific_fn.loc12_36.1(%int_1.loc12) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_36.2: init %i32 = converted %int_1.loc12, %int.convert_checked.loc12_36.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple.elem0.loc12: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc12_36.3: init %i32 = initialize_from %.loc12_36.2 to %tuple.elem0.loc12 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_36.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_36.2: <bound method> = bound_method %int_2, %impl.elem0.loc12_36.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_36.2: <specific function> = specific_function %Convert.bound.loc12_36.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.2: init %i32 = call %Convert.specific_fn.loc12_36.2(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_36.4: init %i32 = converted %int_2, %int.convert_checked.loc12_36.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %v.var, element1
// CHECK:STDOUT:   %.loc12_36.5: init %i32 = initialize_from %.loc12_36.4 to %tuple.elem1.loc12 [template = constants.%int_2.2]
// CHECK:STDOUT:   %impl.elem0.loc12_36.3: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc12_36.3: <bound method> = bound_method %int_3, %impl.elem0.loc12_36.3 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_36.3: <specific function> = specific_function %Convert.bound.loc12_36.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.3: init %i32 = call %Convert.specific_fn.loc12_36.3(%int_3) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc12_36.6: init %i32 = converted %int_3, %int.convert_checked.loc12_36.3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %tuple.elem2.loc12: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc12_36.7: init %i32 = initialize_from %.loc12_36.6 to %tuple.elem2.loc12 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc12_36.8: init %tuple.type.2 = tuple_init (%.loc12_36.3, %.loc12_36.5, %.loc12_36.7) to %v.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_37: init %tuple.type.2 = converted %.loc12_36.1, %.loc12_36.8 [template = constants.%tuple]
// CHECK:STDOUT:   assign %v.var, %.loc12_37
// CHECK:STDOUT:   %int_32.loc14_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_16: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_21: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.1: %tuple.type.1 = tuple_literal (%i32.loc14_11, %i32.loc14_16, %i32.loc14_21)
// CHECK:STDOUT:   %.loc14_24.2: type = converted %.loc14_24.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %v.ref: ref %tuple.type.2 = name_ref v, %v
// CHECK:STDOUT:   %tuple.elem0.loc14: ref %i32 = tuple_access %v.ref, element0
// CHECK:STDOUT:   %.loc14_28.1: %i32 = bind_value %tuple.elem0.loc14
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %v.ref, element1
// CHECK:STDOUT:   %.loc14_28.2: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   %tuple.elem2.loc14: ref %i32 = tuple_access %v.ref, element2
// CHECK:STDOUT:   %.loc14_28.3: %i32 = bind_value %tuple.elem2.loc14
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.loc14_28.1, %.loc14_28.2, %.loc14_28.3)
// CHECK:STDOUT:   %.loc14_29: %tuple.type.2 = converted %v.ref, %tuple
// CHECK:STDOUT:   %w: %tuple.type.2 = bind_name w, %.loc14_29
// CHECK:STDOUT:   %w.ref: %tuple.type.2 = name_ref w, %w
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %tuple.elem1.loc15: %i32 = tuple_access %w.ref, element1
// CHECK:STDOUT:   return %tuple.elem1.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
