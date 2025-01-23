// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/choice/fail_todo_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/choice/fail_todo_params.carbon

// --- fail_todo_params.carbon
library "[[@TEST_NAME]]";

choice C {
  Alt1,
  // CHECK:STDERR: fail_todo_params.carbon:[[@LINE+4]]:7: error: semantics TODO: `choice alternatives with parameters are not yet supported` [SemanticsTodo]
  // CHECK:STDERR:   Alt2(a: i32, b: i64),
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Alt2(a: i32, b: i64),
  Alt3,
}

// --- fail_todo_generic_params.carbon
library "[[@TEST_NAME]]";

choice C(T:! type) {
  // CHECK:STDERR: fail_todo_generic_params.carbon:[[@LINE+4]]:6: error: semantics TODO: `choice alternatives with parameters are not yet supported` [SemanticsTodo]
  // CHECK:STDERR:   Alt(a: T)
  // CHECK:STDERR:      ^~~~~~
  // CHECK:STDERR:
  Alt(a: T)
}

// constants {
//   %S: type = class_type @S [template]
//   %TAlt: type = class_type @TAlt [template]
//   %empty_struct_type: type = struct_type {} [template]
//   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
//   %T.type: type = fn_type @T [template]
//   %T: %T.type = struct_value () [template]
// }
//
// imports {
//   %Core: <namespace> = namespace file.%Core.import, [template] {
//     import Core//prelude
//     import Core//prelude/...
//   }
// }
//
// file {
//   package: <namespace> = namespace [template] {
//     .Core = imports.%Core
//     .S = %S.decl
//   }
//   %Core.import = import Core
//   %S.decl: type = class_decl @S [template = constants.%S] {} {}
// }
//
// class @S {
//   %TAlt.decl: type = class_decl @TAlt [template = constants.%TAlt] {} {}
//   %T.decl: %T.type = fn_decl @T [template = constants.%T] {
//     %return.patt: %S = return_slot_pattern
//     %return.param_patt: %S = out_param_pattern %return.patt, runtime_param0
//   } {
//     %S.ref: type = name_ref S, file.%S.decl [template = constants.%S]
//     %return.param: ref %S = out_param runtime_param0
//     %return: ref %S = return_slot %return.param
//   }
//   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
//
// !members:
//   .Self = constants.%S
//   .TAlt = %TAlt.decl
//   .T = %T.decl
//   complete_type_witness = %complete_type
// }
//
// class @TAlt {
//   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
//
// !members:
//   .Self = constants.%TAlt
//   complete_type_witness = %complete_type
// }
//
// fn @T() -> %S;

// CHECK:STDOUT: --- fail_todo_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %u1: type = class_type @UInt, @UInt(%int_1.5b8) [template]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %u1} [template]
// CHECK:STDOUT:   %complete_type.df6: <witness> = complete_type_witness %struct_type.discriminant [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.5a2: type = fn_type @Convert.1, @ImplicitAs(%u1) [template]
// CHECK:STDOUT:   %impl_witness.514: <witness> = impl_witness (imports.%Core.import_ref.c3d), @impl.1(%int_1.5b8) [template]
// CHECK:STDOUT:   %Convert.type.f9b: type = fn_type @Convert.2, @impl.1(%int_1.5b8) [template]
// CHECK:STDOUT:   %Convert.84e: %Convert.type.f9b = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.db0: <bound method> = bound_method %int_0.5c6, %Convert.84e [template]
// CHECK:STDOUT:   %Convert.specific_fn.e7f: <specific function> = specific_function %Convert.bound.db0, @Convert.2(%int_1.5b8) [template]
// CHECK:STDOUT:   %int_0.cd3: %u1 = int_value 0 [template]
// CHECK:STDOUT:   %C.val.6ef: %C = struct_value (%int_0.cd3) [template]
// CHECK:STDOUT:   %Convert.bound.1f9: <bound method> = bound_method %int_1.5b8, %Convert.84e [template]
// CHECK:STDOUT:   %Convert.specific_fn.9b2: <specific function> = specific_function %Convert.bound.1f9, @Convert.2(%int_1.5b8) [template]
// CHECK:STDOUT:   %int_1.ae4: %u1 = int_value 1 [template]
// CHECK:STDOUT:   %C.val.5bf: %C = struct_value (%int_1.ae4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %a.param: %i32 = value_param runtime_param0
// CHECK:STDOUT:   %.loc9_11: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:   %b.param: %i64 = value_param runtime_param1
// CHECK:STDOUT:   %.loc9_19: type = splice_block %i64 [template = constants.%i64] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [template = constants.%int_64]
// CHECK:STDOUT:     %i64: type = class_type @Int, @Int(constants.%int_64) [template = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %i64 = bind_name b, %b.param
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %u1: type = class_type @UInt, @UInt(constants.%int_1.5b8) [template = constants.%u1]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template = constants.%complete_type.df6]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc4: %Convert.type.5a2 = impl_witness_access constants.%impl_witness.514, element0 [template = constants.%Convert.84e]
// CHECK:STDOUT:   %Convert.bound.loc4: <bound method> = bound_method %int_0, %impl.elem0.loc4 [template = constants.%Convert.bound.db0]
// CHECK:STDOUT:   %Convert.specific_fn.loc4: <specific function> = specific_function %Convert.bound.loc4, @Convert.2(constants.%int_1.5b8) [template = constants.%Convert.specific_fn.e7f]
// CHECK:STDOUT:   %int.convert_checked.loc4: init %u1 = call %Convert.specific_fn.loc4(%int_0) [template = constants.%int_0.cd3]
// CHECK:STDOUT:   %.loc4_7.1: %u1 = value_of_initializer %int.convert_checked.loc4 [template = constants.%int_0.cd3]
// CHECK:STDOUT:   %.loc4_7.2: %u1 = converted %int_0, %.loc4_7.1 [template = constants.%int_0.cd3]
// CHECK:STDOUT:   %.loc4_7.3: %struct_type.discriminant = struct_literal (%.loc4_7.2)
// CHECK:STDOUT:   %.loc4_7.4: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc4_7.5: ref %u1 = class_element_access %.loc4_7.4, element0
// CHECK:STDOUT:   %.loc4_7.6: init %u1 = initialize_from %.loc4_7.2 to %.loc4_7.5 [template = constants.%int_0.cd3]
// CHECK:STDOUT:   %.loc4_7.7: init %C = class_init (%.loc4_7.6), %.loc4_7.4 [template = constants.%C.val.6ef]
// CHECK:STDOUT:   %.loc4_7.8: ref %C = temporary %.loc4_7.4, %.loc4_7.7
// CHECK:STDOUT:   %.loc4_7.9: ref %C = converted %.loc4_7.3, %.loc4_7.8
// CHECK:STDOUT:   %.loc4_7.10: %C = bind_value %.loc4_7.9
// CHECK:STDOUT:   %Alt1: %C = bind_name Alt1, %.loc4_7.10
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc10: %Convert.type.5a2 = impl_witness_access constants.%impl_witness.514, element0 [template = constants.%Convert.84e]
// CHECK:STDOUT:   %Convert.bound.loc10: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10 [template = constants.%Convert.bound.1f9]
// CHECK:STDOUT:   %Convert.specific_fn.loc10: <specific function> = specific_function %Convert.bound.loc10, @Convert.2(constants.%int_1.5b8) [template = constants.%Convert.specific_fn.9b2]
// CHECK:STDOUT:   %int.convert_checked.loc10: init %u1 = call %Convert.specific_fn.loc10(%int_1.loc10) [template = constants.%int_1.ae4]
// CHECK:STDOUT:   %.loc10_7.1: %u1 = value_of_initializer %int.convert_checked.loc10 [template = constants.%int_1.ae4]
// CHECK:STDOUT:   %.loc10_7.2: %u1 = converted %int_1.loc10, %.loc10_7.1 [template = constants.%int_1.ae4]
// CHECK:STDOUT:   %.loc10_7.3: %struct_type.discriminant = struct_literal (%.loc10_7.2)
// CHECK:STDOUT:   %.loc10_7.4: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc10_7.5: ref %u1 = class_element_access %.loc10_7.4, element0
// CHECK:STDOUT:   %.loc10_7.6: init %u1 = initialize_from %.loc10_7.2 to %.loc10_7.5 [template = constants.%int_1.ae4]
// CHECK:STDOUT:   %.loc10_7.7: init %C = class_init (%.loc10_7.6), %.loc10_7.4 [template = constants.%C.val.5bf]
// CHECK:STDOUT:   %.loc10_7.8: ref %C = temporary %.loc10_7.4, %.loc10_7.7
// CHECK:STDOUT:   %.loc10_7.9: ref %C = converted %.loc10_7.3, %.loc10_7.8
// CHECK:STDOUT:   %.loc10_7.10: %C = bind_value %.loc10_7.9
// CHECK:STDOUT:   %Alt3: %C = bind_name Alt3, %.loc10_7.10
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %a
// CHECK:STDOUT:   .b = %b
// CHECK:STDOUT:   .Alt1 = %Alt1
// CHECK:STDOUT:   .Alt3 = %Alt3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_generic_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.discriminant: type = struct_type {.discriminant: %empty_tuple.type} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc3_10.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_10.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc3_10.1, runtime_param<none> [symbolic = %T.patt.loc3_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<none>
// CHECK:STDOUT:     %T.loc3_10.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc3_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc3_10.1: type) {
// CHECK:STDOUT:   %T.loc3_10.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc3_10.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc3_10.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc3_10.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %a.param: @C.%T.loc3_10.2 (%T) = value_param runtime_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc3_10.1 [symbolic = %T.loc3_10.2 (constants.%T)]
// CHECK:STDOUT:     %a: @C.%T.loc3_10.2 (%T) = bind_name a, %a.param
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.discriminant [template = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc3_10.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc3_10.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
