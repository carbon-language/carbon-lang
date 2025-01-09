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
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.dc6: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1aa: type = tuple_type (%tuple.type.dc6, %tuple.type.dc6) [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.e6f: type = tuple_type (%i32, %tuple.type.dc6, %i32) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3b6: type = tuple_type (Core.IntLiteral, %tuple.type.dc6, Core.IntLiteral) [template]
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
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.dc6 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.dc6 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_12: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_17: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_22: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: %tuple.type.ff9 = tuple_literal (%i32.loc11_12, %i32.loc11_17, %i32.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_25.1, constants.%tuple.type.dc6 [template = constants.%tuple.type.dc6]
// CHECK:STDOUT:     %return.param: ref %tuple.type.dc6 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.dc6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.dc6;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %tuple.type.1aa = var v
// CHECK:STDOUT:   %v: ref %tuple.type.1aa = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14_48: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.dc6 = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc14_50: init %tuple.type.dc6 = call %F.ref.loc14_48() to %tuple.elem0
// CHECK:STDOUT:   %F.ref.loc14_53: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.dc6 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc14_55: init %tuple.type.dc6 = call %F.ref.loc14_53() to %tuple.elem1
// CHECK:STDOUT:   %.loc14_56.1: %tuple.type.1aa = tuple_literal (%F.call.loc14_50, %F.call.loc14_55)
// CHECK:STDOUT:   %.loc14_56.2: init %tuple.type.1aa = tuple_init (%F.call.loc14_50, %F.call.loc14_55) to %v.var
// CHECK:STDOUT:   %.loc14_57: init %tuple.type.1aa = converted %.loc14_56.1, %.loc14_56.2
// CHECK:STDOUT:   assign %v.var, %.loc14_57
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref %tuple.type.e6f = var v
// CHECK:STDOUT:   %v: ref %tuple.type.e6f = bind_name v, %v.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.dc6 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %tuple.type.dc6 = call %F.ref() to %tuple.elem1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_50.1: %tuple.type.3b6 = tuple_literal (%int_1, %F.call, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc18_50.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc18_50.1: <bound method> = bound_method %int_1, %impl.elem0.loc18_50.1 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_50.1: <specific function> = specific_function %Convert.bound.loc18_50.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc18_50.1: init %i32 = call %Convert.specific_fn.loc18_50.1(%int_1) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc18_50.2: init %i32 = converted %int_1, %int.convert_checked.loc18_50.1 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc18_50.3: init %i32 = initialize_from %.loc18_50.2 to %tuple.elem0 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc18_50.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc18_50.2: <bound method> = bound_method %int_2, %impl.elem0.loc18_50.2 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_50.2: <specific function> = specific_function %Convert.bound.loc18_50.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc18_50.2: init %i32 = call %Convert.specific_fn.loc18_50.2(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc18_50.4: init %i32 = converted %int_2, %int.convert_checked.loc18_50.2 [template = constants.%int_2.166]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc18_50.5: init %i32 = initialize_from %.loc18_50.4 to %tuple.elem2 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc18_50.6: init %tuple.type.e6f = tuple_init (%.loc18_50.3, %F.call, %.loc18_50.5) to %v.var
// CHECK:STDOUT:   %.loc18_51: init %tuple.type.e6f = converted %.loc18_50.1, %.loc18_50.6
// CHECK:STDOUT:   assign %v.var, %.loc18_51
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
