// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/ir/duplicate_name_same_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/ir/duplicate_name_same_line.carbon

fn A() { if (true) { var n: i32 = 1; } if (true) { var n: i32 = 2; } }

// CHECK:STDOUT: --- duplicate_name_same_line.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
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
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true.loc11_14: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc11_14 br !if.then.loc11_18 else br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_18:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc11_26: %i32 = binding_pattern n
// CHECK:STDOUT:     %.loc11_22.1: %i32 = var_pattern %n.patt.loc11_26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc11_22: ref %i32 = var n
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc11_22: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_22.1: <bound method> = bound_method %int_1, %impl.elem0.loc11_22 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc11_22: <specific function> = specific_function %impl.elem0.loc11_22, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_22.2: <bound method> = bound_method %int_1, %specific_fn.loc11_22 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc11_22: init %i32 = call %bound_method.loc11_22.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_22.2: init %i32 = converted %int_1, %int.convert_checked.loc11_22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %n.var.loc11_22, %.loc11_22.2
// CHECK:STDOUT:   %.loc11_29: type = splice_block %i32.loc11_29 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc11_26: ref %i32 = bind_name n, %n.var.loc11_22
// CHECK:STDOUT:   br !if.else.loc11_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_18:
// CHECK:STDOUT:   %true.loc11_44: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true.loc11_44 br !if.then.loc11_48 else br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11_48:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt.loc11_56: %i32 = binding_pattern n
// CHECK:STDOUT:     %.loc11_52.1: %i32 = var_pattern %n.patt.loc11_56
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var.loc11_52: ref %i32 = var n
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc11_52: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_52.1: <bound method> = bound_method %int_2, %impl.elem0.loc11_52 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc11_52: <specific function> = specific_function %impl.elem0.loc11_52, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_52.2: <bound method> = bound_method %int_2, %specific_fn.loc11_52 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc11_52: init %i32 = call %bound_method.loc11_52.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_52.2: init %i32 = converted %int_2, %int.convert_checked.loc11_52 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   assign %n.var.loc11_52, %.loc11_52.2
// CHECK:STDOUT:   %.loc11_59: type = splice_block %i32.loc11_59 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11_59: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_59: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.loc11_56: ref %i32 = bind_name n, %n.var.loc11_52
// CHECK:STDOUT:   br !if.else.loc11_48
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11_48:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
