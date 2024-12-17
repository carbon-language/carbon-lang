// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: ((i32, i32, i32), (i32, i32, i32)) = (F(), F());
}

fn H() {
  var v: (i32, (i32, i32, i32), i32) = (1, F(), 2);
}

// CHECK:STDOUT: --- nested_tuple_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%tuple.type.1, %tuple.type.1) [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (%tuple.type.2, %tuple.type.2) [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.6: type = tuple_type (type, %tuple.type.1, type) [template]
// CHECK:STDOUT:   %tuple.type.7: type = tuple_type (%i32, %tuple.type.2, %i32) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.9: type = tuple_type (Core.IntLiteral, %tuple.type.2, Core.IntLiteral) [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: %tuple.type.1 = tuple_literal (%i32.loc11_12, %i32.loc11_17, %i32.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_25.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc14_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25: %tuple.type.1 = tuple_literal (%i32.loc14_12, %i32.loc14_17, %i32.loc14_22)
// CHECK:STDOUT:   %int_32.loc14_29: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_29: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_34: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_34: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_39: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_39: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_42: %tuple.type.1 = tuple_literal (%i32.loc14_29, %i32.loc14_34, %i32.loc14_39)
// CHECK:STDOUT:   %.loc14_43.1: %tuple.type.3 = tuple_literal (%.loc14_25, %.loc14_42)
// CHECK:STDOUT:   %.loc14_43.2: type = converted %.loc14_25, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc14_43.3: type = converted %.loc14_42, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc14_43.4: type = converted %.loc14_43.1, constants.%tuple.type.4 [template = constants.%tuple.type.4]
// CHECK:STDOUT:   %v.var: ref %tuple.type.4 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.4 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14_48: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.2 = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc14_50: init %tuple.type.2 = call %F.ref.loc14_48() to %tuple.elem0
// CHECK:STDOUT:   %F.ref.loc14_53: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.2 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc14_55: init %tuple.type.2 = call %F.ref.loc14_53() to %tuple.elem1
// CHECK:STDOUT:   %.loc14_56.1: %tuple.type.4 = tuple_literal (%F.call.loc14_50, %F.call.loc14_55)
// CHECK:STDOUT:   %.loc14_56.2: init %tuple.type.4 = tuple_init (%F.call.loc14_50, %F.call.loc14_55) to %v.var
// CHECK:STDOUT:   %.loc14_57: init %tuple.type.4 = converted %.loc14_56.1, %.loc14_56.2
// CHECK:STDOUT:   assign %v.var, %.loc14_57
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc18_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_11: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18_27: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_27: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_30: %tuple.type.1 = tuple_literal (%i32.loc18_17, %i32.loc18_22, %i32.loc18_27)
// CHECK:STDOUT:   %int_32.loc18_33: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc18_33: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_36.1: %tuple.type.6 = tuple_literal (%i32.loc18_11, %.loc18_30, %i32.loc18_33)
// CHECK:STDOUT:   %.loc18_36.2: type = converted %.loc18_30, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc18_36.3: type = converted %.loc18_36.1, constants.%tuple.type.7 [template = constants.%tuple.type.7]
// CHECK:STDOUT:   %v.var: ref %tuple.type.7 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.7 = bind_name v, %v.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.2 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %tuple.type.2 = call %F.ref() to %tuple.elem1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc18_50.1: %tuple.type.9 = tuple_literal (%int_1, %F.call, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc18_50.1: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_50.1: <bound method> = bound_method %int_1, %impl.elem0.loc18_50.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_50.1: <specific function> = specific_function %Convert.bound.loc18_50.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_50.1: init %i32 = call %Convert.specific_fn.loc18_50.1(%int_1) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_50.2: init %i32 = converted %int_1, %int.convert_checked.loc18_50.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc18_50.3: init %i32 = initialize_from %.loc18_50.2 to %tuple.elem0 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc18_50.2: %Convert.type.2 = interface_witness_access constants.%interface.5, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:   %Convert.bound.loc18_50.2: <bound method> = bound_method %int_2, %impl.elem0.loc18_50.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_50.2: <specific function> = specific_function %Convert.bound.loc18_50.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc18_50.2: init %i32 = call %Convert.specific_fn.loc18_50.2(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc18_50.4: init %i32 = converted %int_2, %int.convert_checked.loc18_50.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc18_50.5: init %i32 = initialize_from %.loc18_50.4 to %tuple.elem2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc18_50.6: init %tuple.type.7 = tuple_init (%.loc18_50.3, %F.call, %.loc18_50.5) to %v.var
// CHECK:STDOUT:   %.loc18_51: init %tuple.type.7 = converted %.loc18_50.1, %.loc18_50.6
// CHECK:STDOUT:   assign %v.var, %.loc18_51
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
