// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/index/expr_category.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/index/expr_category.carbon

fn F() -> [i32; 3];

fn G(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Indexing a durable array reference gives a durable reference.
  var pa: i32* = &a[0];
  a[0] = 4;
}

fn ValueBinding(b: [i32; 3]) {
  var a: [i32; 3] = (1, 2, 3);

  // Index but don't do anything else so we can check that a value binding is
  // produced when appropriate.
  a[0];
  b[0];
  F()[0];
}

// CHECK:STDOUT: --- expr_category.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, i32 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.5, %.6, %.2) [template]
// CHECK:STDOUT:   %.9: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %ValueBinding.type: type = fn_type @ValueBinding [template]
// CHECK:STDOUT:   %ValueBinding: %ValueBinding.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .ValueBinding = %ValueBinding.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_17: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_12.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_12.2: type = converted %int.make_type_32.loc11, %.loc11_12.1 [template = i32]
// CHECK:STDOUT:     %.loc11_18: type = array_type %.loc11_17, i32 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref %.3 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_15: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc13_10.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_10.2: type = converted %int.make_type_32.loc13, %.loc13_10.1 [template = i32]
// CHECK:STDOUT:     %.loc13_16: type = array_type %.loc13_15, i32 [template = constants.%.3]
// CHECK:STDOUT:     %b.loc13_6.1: %.3 = param b, runtime_param0
// CHECK:STDOUT:     @G.%b: %.3 = bind_name b, %b.loc13_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ValueBinding.decl: %ValueBinding.type = fn_decl @ValueBinding [template = constants.%ValueBinding] {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc21_26: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:     %.loc21_21.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_21.2: type = converted %int.make_type_32.loc21, %.loc21_21.1 [template = i32]
// CHECK:STDOUT:     %.loc21_27: type = array_type %.loc21_26, i32 [template = constants.%.3]
// CHECK:STDOUT:     %b.loc21_17.1: %.3 = param b, runtime_param0
// CHECK:STDOUT:     @ValueBinding.%b: %.3 = bind_name b, %b.loc21_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_16: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_11.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_11.2: type = converted %int.make_type_32.loc14, %.loc14_11.1 [template = i32]
// CHECK:STDOUT:   %.loc14_17: type = array_type %.loc14_16, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc14_22: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_25: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_28: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_29.1: %.7 = tuple_literal (%.loc14_22, %.loc14_25, %.loc14_28)
// CHECK:STDOUT:   %.loc14_29.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc14_29.3: ref i32 = array_index %a.var, %.loc14_29.2
// CHECK:STDOUT:   %.loc14_29.4: init i32 = initialize_from %.loc14_22 to %.loc14_29.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_29.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc14_29.6: ref i32 = array_index %a.var, %.loc14_29.5
// CHECK:STDOUT:   %.loc14_29.7: init i32 = initialize_from %.loc14_25 to %.loc14_29.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_29.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc14_29.9: ref i32 = array_index %a.var, %.loc14_29.8
// CHECK:STDOUT:   %.loc14_29.10: init i32 = initialize_from %.loc14_28 to %.loc14_29.9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_29.11: init %.3 = array_init (%.loc14_29.4, %.loc14_29.7, %.loc14_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc14_30: init %.3 = converted %.loc14_29.1, %.loc14_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc14_30
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_14.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_14.2: type = converted %int.make_type_32.loc17, %.loc17_14.1 [template = i32]
// CHECK:STDOUT:   %.loc17_14.3: type = ptr_type i32 [template = constants.%.9]
// CHECK:STDOUT:   %pa.var: ref %.9 = var pa
// CHECK:STDOUT:   %pa: ref %.9 = bind_name pa, %pa.var
// CHECK:STDOUT:   %a.ref.loc17: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc17_21: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc17_22: ref i32 = array_index %a.ref.loc17, %.loc17_21
// CHECK:STDOUT:   %.loc17_18: %.9 = addr_of %.loc17_22
// CHECK:STDOUT:   assign %pa.var, %.loc17_18
// CHECK:STDOUT:   %a.ref.loc18: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc18_5: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc18_6: ref i32 = array_index %a.ref.loc18, %.loc18_5
// CHECK:STDOUT:   %.loc18_10: i32 = int_literal 4 [template = constants.%.10]
// CHECK:STDOUT:   assign %.loc18_6, %.loc18_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ValueBinding(%b: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc22_16: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc22_11.2: type = converted %int.make_type_32, %.loc22_11.1 [template = i32]
// CHECK:STDOUT:   %.loc22_17: type = array_type %.loc22_16, i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc22_22: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_25: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc22_28: i32 = int_literal 3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_29.1: %.7 = tuple_literal (%.loc22_22, %.loc22_25, %.loc22_28)
// CHECK:STDOUT:   %.loc22_29.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc22_29.3: ref i32 = array_index %a.var, %.loc22_29.2
// CHECK:STDOUT:   %.loc22_29.4: init i32 = initialize_from %.loc22_22 to %.loc22_29.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_29.5: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc22_29.6: ref i32 = array_index %a.var, %.loc22_29.5
// CHECK:STDOUT:   %.loc22_29.7: init i32 = initialize_from %.loc22_25 to %.loc22_29.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc22_29.8: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc22_29.9: ref i32 = array_index %a.var, %.loc22_29.8
// CHECK:STDOUT:   %.loc22_29.10: init i32 = initialize_from %.loc22_28 to %.loc22_29.9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_29.11: init %.3 = array_init (%.loc22_29.4, %.loc22_29.7, %.loc22_29.10) to %a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc22_30: init %.3 = converted %.loc22_29.1, %.loc22_29.11 [template = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc22_30
// CHECK:STDOUT:   %a.ref: ref %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc26_5: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc26_6: ref i32 = array_index %a.ref, %.loc26_5
// CHECK:STDOUT:   %b.ref: %.3 = name_ref b, %b
// CHECK:STDOUT:   %.loc27_5: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc27_6.1: ref %.3 = value_as_ref %b.ref
// CHECK:STDOUT:   %.loc27_6.2: ref i32 = array_index %.loc27_6.1, %.loc27_5
// CHECK:STDOUT:   %.loc27_6.3: i32 = bind_value %.loc27_6.2
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc28_4.1: ref %.3 = temporary_storage
// CHECK:STDOUT:   %F.call: init %.3 = call %F.ref() to %.loc28_4.1
// CHECK:STDOUT:   %.loc28_7: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc28_4.2: ref %.3 = temporary %.loc28_4.1, %F.call
// CHECK:STDOUT:   %.loc28_8.1: ref i32 = array_index %.loc28_4.2, %.loc28_7
// CHECK:STDOUT:   %.loc28_8.2: i32 = bind_value %.loc28_8.1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
