// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/address_of_lvalue.carbon

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};

  var p: {.a: i32, .b: i32}* = &s;
  var q: i32* = &s.a;
  var r: i32* = &s.b;

  var t: (i32, i32) = (1, 2);
  var t0: i32* = &t.0;
  var t1: i32* = &t.1;
}

// CHECK:STDOUT: --- address_of_lvalue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %.3: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%.3, %.4) [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type %tuple.type.2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.3, %.4) [template]
// CHECK:STDOUT:   %.7: i32 = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc12_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_15.1: type = value_of_initializer %int.make_type_32.loc12_15 [template = i32]
// CHECK:STDOUT:   %.loc12_15.2: type = converted %int.make_type_32.loc12_15, %.loc12_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc12_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_24.1: type = value_of_initializer %int.make_type_32.loc12_24 [template = i32]
// CHECK:STDOUT:   %.loc12_24.2: type = converted %int.make_type_32.loc12_24, %.loc12_24.1 [template = i32]
// CHECK:STDOUT:   %.loc12_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %s.var: ref %.1 = var s
// CHECK:STDOUT:   %s: ref %.1 = bind_name s, %s.var
// CHECK:STDOUT:   %.loc12_37: i32 = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_45: i32 = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_46.1: %.1 = struct_literal (%.loc12_37, %.loc12_45)
// CHECK:STDOUT:   %.loc12_46.2: ref i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc12_46.3: init i32 = initialize_from %.loc12_37 to %.loc12_46.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc12_46.4: ref i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc12_46.5: init i32 = initialize_from %.loc12_45 to %.loc12_46.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_46.6: init %.1 = struct_init (%.loc12_46.3, %.loc12_46.5) to %s.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_47: init %.1 = converted %.loc12_46.1, %.loc12_46.6 [template = constants.%struct]
// CHECK:STDOUT:   assign %s.var, %.loc12_47
// CHECK:STDOUT:   %int.make_type_32.loc14_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_15.1: type = value_of_initializer %int.make_type_32.loc14_15 [template = i32]
// CHECK:STDOUT:   %.loc14_15.2: type = converted %int.make_type_32.loc14_15, %.loc14_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_24.1: type = value_of_initializer %int.make_type_32.loc14_24 [template = i32]
// CHECK:STDOUT:   %.loc14_24.2: type = converted %int.make_type_32.loc14_24, %.loc14_24.1 [template = i32]
// CHECK:STDOUT:   %.loc14_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_28: type = ptr_type %.1 [template = constants.%.2]
// CHECK:STDOUT:   %p.var: ref %.2 = var p
// CHECK:STDOUT:   %p: ref %.2 = bind_name p, %p.var
// CHECK:STDOUT:   %s.ref.loc14: ref %.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc14_32: %.2 = addr_of %s.ref.loc14
// CHECK:STDOUT:   assign %p.var, %.loc14_32
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_13.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_13.2: type = converted %int.make_type_32.loc15, %.loc15_13.1 [template = i32]
// CHECK:STDOUT:   %.loc15_13.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:   %q.var: ref %.5 = var q
// CHECK:STDOUT:   %q: ref %.5 = bind_name q, %q.var
// CHECK:STDOUT:   %s.ref.loc15: ref %.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc15_19: ref i32 = struct_access %s.ref.loc15, element0
// CHECK:STDOUT:   %.loc15_17: %.5 = addr_of %.loc15_19
// CHECK:STDOUT:   assign %q.var, %.loc15_17
// CHECK:STDOUT:   %int.make_type_32.loc16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_13.1: type = value_of_initializer %int.make_type_32.loc16 [template = i32]
// CHECK:STDOUT:   %.loc16_13.2: type = converted %int.make_type_32.loc16, %.loc16_13.1 [template = i32]
// CHECK:STDOUT:   %.loc16_13.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:   %r.var: ref %.5 = var r
// CHECK:STDOUT:   %r: ref %.5 = bind_name r, %r.var
// CHECK:STDOUT:   %s.ref.loc16: ref %.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc16_19: ref i32 = struct_access %s.ref.loc16, element1
// CHECK:STDOUT:   %.loc16_17: %.5 = addr_of %.loc16_19
// CHECK:STDOUT:   assign %r.var, %.loc16_17
// CHECK:STDOUT:   %int.make_type_32.loc18_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc18_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc18_19.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc18_11, %int.make_type_32.loc18_16)
// CHECK:STDOUT:   %.loc18_19.2: type = value_of_initializer %int.make_type_32.loc18_11 [template = i32]
// CHECK:STDOUT:   %.loc18_19.3: type = converted %int.make_type_32.loc18_11, %.loc18_19.2 [template = i32]
// CHECK:STDOUT:   %.loc18_19.4: type = value_of_initializer %int.make_type_32.loc18_16 [template = i32]
// CHECK:STDOUT:   %.loc18_19.5: type = converted %int.make_type_32.loc18_16, %.loc18_19.4 [template = i32]
// CHECK:STDOUT:   %.loc18_19.6: type = converted %.loc18_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %t.var: ref %tuple.type.2 = var t
// CHECK:STDOUT:   %t: ref %tuple.type.2 = bind_name t, %t.var
// CHECK:STDOUT:   %.loc18_24: i32 = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc18_27: i32 = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_28.1: %tuple.type.2 = tuple_literal (%.loc18_24, %.loc18_27)
// CHECK:STDOUT:   %.loc18_28.2: ref i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc18_28.3: init i32 = initialize_from %.loc18_24 to %.loc18_28.2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc18_28.4: ref i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc18_28.5: init i32 = initialize_from %.loc18_27 to %.loc18_28.4 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_28.6: init %tuple.type.2 = tuple_init (%.loc18_28.3, %.loc18_28.5) to %t.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc18_29: init %tuple.type.2 = converted %.loc18_28.1, %.loc18_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %t.var, %.loc18_29
// CHECK:STDOUT:   %int.make_type_32.loc19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_14.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:   %.loc19_14.2: type = converted %int.make_type_32.loc19, %.loc19_14.1 [template = i32]
// CHECK:STDOUT:   %.loc19_14.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:   %t0.var: ref %.5 = var t0
// CHECK:STDOUT:   %t0: ref %.5 = bind_name t0, %t0.var
// CHECK:STDOUT:   %t.ref.loc19: ref %tuple.type.2 = name_ref t, %t
// CHECK:STDOUT:   %.loc19_21: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc19_20: ref i32 = tuple_access %t.ref.loc19, element0
// CHECK:STDOUT:   %.loc19_18: %.5 = addr_of %.loc19_20
// CHECK:STDOUT:   assign %t0.var, %.loc19_18
// CHECK:STDOUT:   %int.make_type_32.loc20: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc20_14.1: type = value_of_initializer %int.make_type_32.loc20 [template = i32]
// CHECK:STDOUT:   %.loc20_14.2: type = converted %int.make_type_32.loc20, %.loc20_14.1 [template = i32]
// CHECK:STDOUT:   %.loc20_14.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:   %t1.var: ref %.5 = var t1
// CHECK:STDOUT:   %t1: ref %.5 = bind_name t1, %t1.var
// CHECK:STDOUT:   %t.ref.loc20: ref %tuple.type.2 = name_ref t, %t
// CHECK:STDOUT:   %.loc20_21: i32 = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc20_20: ref i32 = tuple_access %t.ref.loc20, element1
// CHECK:STDOUT:   %.loc20_18: %.5 = addr_of %.loc20_20
// CHECK:STDOUT:   assign %t1.var, %.loc20_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
