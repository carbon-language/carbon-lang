// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon

fn Main() {
  var a: i32 = 12;
  a = 9;

  var b: (i32, i32) = (1, 2);
  b.0 = 3;
  b.1 = 4;

  var c: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  c.a = 3;
  c.b = 4;

  var p: i32* = &a;
  *p = 5;

  *(if true then p else &a) = 10;
}

// CHECK:STDOUT: --- assignment.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %tuple: %.5 = tuple_value (%.7, %.8) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.12: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.13: type = ptr_type %.12 [template]
// CHECK:STDOUT:   %struct: %.12 = struct_value (%.7, %.8) [template]
// CHECK:STDOUT:   %.14: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.16: bool = bool_literal true [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32.loc12, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 12 [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc12_16
// CHECK:STDOUT:   %a.ref.loc13: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc13: i32 = int_literal 9 [template = constants.%.3]
// CHECK:STDOUT:   assign %a.ref.loc13, %.loc13
// CHECK:STDOUT:   %int.make_type_32.loc15_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc15_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_19.1: %.4 = tuple_literal (%int.make_type_32.loc15_11, %int.make_type_32.loc15_16)
// CHECK:STDOUT:   %.loc15_19.2: type = value_of_initializer %int.make_type_32.loc15_11 [template = i32]
// CHECK:STDOUT:   %.loc15_19.3: type = converted %int.make_type_32.loc15_11, %.loc15_19.2 [template = i32]
// CHECK:STDOUT:   %.loc15_19.4: type = value_of_initializer %int.make_type_32.loc15_16 [template = i32]
// CHECK:STDOUT:   %.loc15_19.5: type = converted %int.make_type_32.loc15_16, %.loc15_19.4 [template = i32]
// CHECK:STDOUT:   %.loc15_19.6: type = converted %.loc15_19.1, constants.%.5 [template = constants.%.5]
// CHECK:STDOUT:   %b.var: ref %.5 = var b
// CHECK:STDOUT:   %b: ref %.5 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc15_24: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_27: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc15_28.1: %.5 = tuple_literal (%.loc15_24, %.loc15_27)
// CHECK:STDOUT:   %.loc15_28.2: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc15_28.3: init i32 = initialize_from %.loc15_24 to %.loc15_28.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_28.4: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc15_28.5: init i32 = initialize_from %.loc15_27 to %.loc15_28.4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc15_28.6: init %.5 = tuple_init (%.loc15_28.3, %.loc15_28.5) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_29: init %.5 = converted %.loc15_28.1, %.loc15_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc15_29
// CHECK:STDOUT:   %b.ref.loc20: ref %.5 = name_ref b, %b
// CHECK:STDOUT:   %.loc20_5: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:   %.loc20_4: ref i32 = tuple_index %b.ref.loc20, %.loc20_5
// CHECK:STDOUT:   %.loc20_9: i32 = int_literal 3 [template = constants.%.10]
// CHECK:STDOUT:   assign %.loc20_4, %.loc20_9
// CHECK:STDOUT:   %b.ref.loc24: ref %.5 = name_ref b, %b
// CHECK:STDOUT:   %.loc24_5: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc24_4: ref i32 = tuple_index %b.ref.loc24, %.loc24_5
// CHECK:STDOUT:   %.loc24_9: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   assign %.loc24_4, %.loc24_9
// CHECK:STDOUT:   %int.make_type_32.loc26_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_15.1: type = value_of_initializer %int.make_type_32.loc26_15 [template = i32]
// CHECK:STDOUT:   %.loc26_15.2: type = converted %int.make_type_32.loc26_15, %.loc26_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc26_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc26_24.1: type = value_of_initializer %int.make_type_32.loc26_24 [template = i32]
// CHECK:STDOUT:   %.loc26_24.2: type = converted %int.make_type_32.loc26_24, %.loc26_24.1 [template = i32]
// CHECK:STDOUT:   %.loc26_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.12]
// CHECK:STDOUT:   %c.var: ref %.12 = var c
// CHECK:STDOUT:   %c: ref %.12 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc26_37: i32 = int_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc26_45: i32 = int_literal 2 [template = constants.%.8]
// CHECK:STDOUT:   %.loc26_46.1: %.12 = struct_literal (%.loc26_37, %.loc26_45)
// CHECK:STDOUT:   %.loc26_46.2: ref i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc26_46.3: init i32 = initialize_from %.loc26_37 to %.loc26_46.2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc26_46.4: ref i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc26_46.5: init i32 = initialize_from %.loc26_45 to %.loc26_46.4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc26_46.6: init %.12 = struct_init (%.loc26_46.3, %.loc26_46.5) to %c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc26_47: init %.12 = converted %.loc26_46.1, %.loc26_46.6 [template = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc26_47
// CHECK:STDOUT:   %c.ref.loc27: ref %.12 = name_ref c, %c
// CHECK:STDOUT:   %.loc27_4: ref i32 = struct_access %c.ref.loc27, element0
// CHECK:STDOUT:   %.loc27_9: i32 = int_literal 3 [template = constants.%.10]
// CHECK:STDOUT:   assign %.loc27_4, %.loc27_9
// CHECK:STDOUT:   %c.ref.loc28: ref %.12 = name_ref c, %c
// CHECK:STDOUT:   %.loc28_4: ref i32 = struct_access %c.ref.loc28, element1
// CHECK:STDOUT:   %.loc28_9: i32 = int_literal 4 [template = constants.%.11]
// CHECK:STDOUT:   assign %.loc28_4, %.loc28_9
// CHECK:STDOUT:   %int.make_type_32.loc30: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc30_13.1: type = value_of_initializer %int.make_type_32.loc30 [template = i32]
// CHECK:STDOUT:   %.loc30_13.2: type = converted %int.make_type_32.loc30, %.loc30_13.1 [template = i32]
// CHECK:STDOUT:   %.loc30_13.3: type = ptr_type i32 [template = constants.%.14]
// CHECK:STDOUT:   %p.var: ref %.14 = var p
// CHECK:STDOUT:   %p: ref %.14 = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref.loc30: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc30_17: %.14 = addr_of %a.ref.loc30
// CHECK:STDOUT:   assign %p.var, %.loc30_17
// CHECK:STDOUT:   %p.ref.loc31: ref %.14 = name_ref p, %p
// CHECK:STDOUT:   %.loc31_4: %.14 = bind_value %p.ref.loc31
// CHECK:STDOUT:   %.loc31_3: ref i32 = deref %.loc31_4
// CHECK:STDOUT:   %.loc31_8: i32 = int_literal 5 [template = constants.%.15]
// CHECK:STDOUT:   assign %.loc31_3, %.loc31_8
// CHECK:STDOUT:   %.loc33_8: bool = bool_literal true [template = constants.%.16]
// CHECK:STDOUT:   if %.loc33_8 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc33: ref %.14 = name_ref p, %p
// CHECK:STDOUT:   %.loc33_18: %.14 = bind_value %p.ref.loc33
// CHECK:STDOUT:   br !if.expr.result(%.loc33_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc33: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc33_25: %.14 = addr_of %a.ref.loc33
// CHECK:STDOUT:   br !if.expr.result(%.loc33_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc33_5: %.14 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc33_3: ref i32 = deref %.loc33_5
// CHECK:STDOUT:   %.loc33_31: i32 = int_literal 10 [template = constants.%.17]
// CHECK:STDOUT:   assign %.loc33_3, %.loc33_31
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
