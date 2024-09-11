// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/constant_condition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/constant_condition.carbon

fn A() -> i32 { return 1; }
fn B() -> i32 { return 2; }

fn F() -> i32 {
  return if true then A() else B();
}

fn G() -> i32 {
  return if false then A() else B();
}

fn Constant() -> i32 {
  var v: if true then i32 else i32* = 1;
  var w: if false then i32 else i32* = &v;
  return *w;
}

fn PartiallyConstant(t: type) -> i32 {
  var v: if true then i32 else t = 1;
  var w: if false then t else i32* = &v;
  return *w;
}

// CHECK:STDOUT: --- constant_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %B.type: type = fn_type @B [template]
// CHECK:STDOUT:   %B: %B.type = struct_value () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: bool = bool_literal true [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: bool = bool_literal false [template]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [template]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = ptr_type i32 [template]
// CHECK:STDOUT:   %PartiallyConstant.type: type = fn_type @PartiallyConstant [template]
// CHECK:STDOUT:   %PartiallyConstant: %PartiallyConstant.type = struct_value () [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartiallyConstant = %PartiallyConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %int.make_type_32.loc11, %.loc11_11.1 [template = i32]
// CHECK:STDOUT:     @A.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [template = constants.%B] {
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %int.make_type_32.loc12, %.loc12_11.1 [template = i32]
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_11.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:     %.loc14_11.2: type = converted %int.make_type_32.loc14, %.loc14_11.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_11.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:     %.loc18_11.2: type = converted %int.make_type_32.loc18, %.loc18_11.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [template = constants.%Constant] {
// CHECK:STDOUT:     %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc22_18.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:     %.loc22_18.2: type = converted %int.make_type_32.loc22, %.loc22_18.1 [template = i32]
// CHECK:STDOUT:     @Constant.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PartiallyConstant.decl: %PartiallyConstant.type = fn_decl @PartiallyConstant [template = constants.%PartiallyConstant] {
// CHECK:STDOUT:     %t.loc28_22.1: type = param t
// CHECK:STDOUT:     @PartiallyConstant.%t: type = bind_name t, %t.loc28_22.1
// CHECK:STDOUT:     %int.make_type_32.loc28: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc28_34.1: type = value_of_initializer %int.make_type_32.loc28 [template = i32]
// CHECK:STDOUT:     %.loc28_34.2: type = converted %int.make_type_32.loc28, %.loc28_34.1 [template = i32]
// CHECK:STDOUT:     @PartiallyConstant.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_literal true [template = constants.%.4]
// CHECK:STDOUT:   if %.loc15_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc15_24.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc15_24.2: i32 = converted %A.call, %.loc15_24.1
// CHECK:STDOUT:   br !if.expr.result(%.loc15_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc15_27.1: i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc15_27.2: i32 = converted %B.call, %.loc15_27.1
// CHECK:STDOUT:   br !if.expr.result(%.loc15_27.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc15_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc15_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc19_13: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc19_13 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %A.call: init i32 = call %A.ref()
// CHECK:STDOUT:   %.loc19_25.1: i32 = value_of_initializer %A.call
// CHECK:STDOUT:   %.loc19_25.2: i32 = converted %A.call, %.loc19_25.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %B.ref: %B.type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %B.call: init i32 = call %B.ref()
// CHECK:STDOUT:   %.loc19_28.1: i32 = value_of_initializer %B.call
// CHECK:STDOUT:   %.loc19_28.2: i32 = converted %B.call, %.loc19_28.1
// CHECK:STDOUT:   br !if.expr.result(%.loc19_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc19_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc19_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_13: bool = bool_literal true [template = constants.%.4]
// CHECK:STDOUT:   if %.loc23_13 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %int.make_type_32.loc23_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_23.1: type = value_of_initializer %int.make_type_32.loc23_23 [template = i32]
// CHECK:STDOUT:   %.loc23_23.2: type = converted %int.make_type_32.loc23_23, %.loc23_23.1 [template = i32]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_23.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %int.make_type_32.loc23_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_35.1: type = value_of_initializer %int.make_type_32.loc23_32 [template = i32]
// CHECK:STDOUT:   %.loc23_35.2: type = converted %int.make_type_32.loc23_32, %.loc23_35.1 [template = i32]
// CHECK:STDOUT:   %.loc23_35.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_35.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_10: type = block_arg !if.expr.result.loc23 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc23_39: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc23_39
// CHECK:STDOUT:   %.loc24_13: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc24_13 br !if.expr.then.loc24 else br !if.expr.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc24:
// CHECK:STDOUT:   %int.make_type_32.loc24_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc24_24.1: type = value_of_initializer %int.make_type_32.loc24_24 [template = i32]
// CHECK:STDOUT:   %.loc24_24.2: type = converted %int.make_type_32.loc24_24, %.loc24_24.1 [template = i32]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc24:
// CHECK:STDOUT:   %int.make_type_32.loc24_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc24_36.1: type = value_of_initializer %int.make_type_32.loc24_33 [template = i32]
// CHECK:STDOUT:   %.loc24_36.2: type = converted %int.make_type_32.loc24_33, %.loc24_36.1 [template = i32]
// CHECK:STDOUT:   %.loc24_36.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_36.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc24:
// CHECK:STDOUT:   %.loc24_10: type = block_arg !if.expr.result.loc24 [template = constants.%.6]
// CHECK:STDOUT:   %w.var: ref %.6 = var w
// CHECK:STDOUT:   %w: ref %.6 = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc24_40: %.6 = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %.loc24_40
// CHECK:STDOUT:   %w.ref: ref %.6 = name_ref w, %w
// CHECK:STDOUT:   %.loc25_11: %.6 = bind_value %w.ref
// CHECK:STDOUT:   %.loc25_10.1: ref i32 = deref %.loc25_11
// CHECK:STDOUT:   %.loc25_10.2: i32 = bind_value %.loc25_10.1
// CHECK:STDOUT:   return %.loc25_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartiallyConstant(%t: type) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29_13: bool = bool_literal true [template = constants.%.4]
// CHECK:STDOUT:   if %.loc29_13 br !if.expr.then.loc29 else br !if.expr.else.loc29
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc29:
// CHECK:STDOUT:   %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc29_23.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:   %.loc29_23.2: type = converted %int.make_type_32.loc29, %.loc29_23.1 [template = i32]
// CHECK:STDOUT:   br !if.expr.result.loc29(%.loc29_23.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc29:
// CHECK:STDOUT:   %t.ref.loc29: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc29(%t.ref.loc29)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc29:
// CHECK:STDOUT:   %.loc29_10: type = block_arg !if.expr.result.loc29 [template = i32]
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc29_36: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc29_36
// CHECK:STDOUT:   %.loc30_13: bool = bool_literal false [template = constants.%.5]
// CHECK:STDOUT:   if %.loc30_13 br !if.expr.then.loc30 else br !if.expr.else.loc30
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc30:
// CHECK:STDOUT:   %t.ref.loc30: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc30(%t.ref.loc30)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc30:
// CHECK:STDOUT:   %int.make_type_32.loc30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc30_34.1: type = value_of_initializer %int.make_type_32.loc30 [template = i32]
// CHECK:STDOUT:   %.loc30_34.2: type = converted %int.make_type_32.loc30, %.loc30_34.1 [template = i32]
// CHECK:STDOUT:   %.loc30_34.3: type = ptr_type i32 [template = constants.%.6]
// CHECK:STDOUT:   br !if.expr.result.loc30(%.loc30_34.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc30:
// CHECK:STDOUT:   %.loc30_10: type = block_arg !if.expr.result.loc30 [template = constants.%.6]
// CHECK:STDOUT:   %w.var: ref %.6 = var w
// CHECK:STDOUT:   %w: ref %.6 = bind_name w, %w.var
// CHECK:STDOUT:   %v.ref: ref i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc30_38: %.6 = addr_of %v.ref
// CHECK:STDOUT:   assign %w.var, %.loc30_38
// CHECK:STDOUT:   %w.ref: ref %.6 = name_ref w, %w
// CHECK:STDOUT:   %.loc31_11: %.6 = bind_value %w.ref
// CHECK:STDOUT:   %.loc31_10.1: ref i32 = deref %.loc31_11
// CHECK:STDOUT:   %.loc31_10.2: i32 = bind_value %.loc31_10.1
// CHECK:STDOUT:   return %.loc31_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
