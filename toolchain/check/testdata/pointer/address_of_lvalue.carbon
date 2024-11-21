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
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value (%.33, %.36) [template]
// CHECK:STDOUT:   %.37: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.33, %.36) [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_15: init type = call constants.%Int(%.loc12_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.2: type = value_of_initializer %int.make_type_signed.loc12_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.3: type = converted %int.make_type_signed.loc12_15, %.loc12_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_24: init type = call constants.%Int(%.loc12_24.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.2: type = value_of_initializer %int.make_type_signed.loc12_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.3: type = converted %int.make_type_signed.loc12_24, %.loc12_24.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_27: type = struct_type {.a: %i32, .b: %i32} [template = constants.%.2]
// CHECK:STDOUT:   %s.var: ref %.2 = var s
// CHECK:STDOUT:   %s: ref %.2 = bind_name s, %s.var
// CHECK:STDOUT:   %.loc12_37: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_45: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc12_46.1: %.6 = struct_literal (%.loc12_37, %.loc12_45)
// CHECK:STDOUT:   %.loc12_46.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_46.3: <bound method> = bound_method %.loc12_37, %.loc12_46.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_46.4: <specific function> = specific_function %.loc12_46.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc12_46.1: init %i32 = call %.loc12_46.4(%.loc12_37) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_46.5: init %i32 = converted %.loc12_37, %int.convert_checked.loc12_46.1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_46.6: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc12_46.7: init %i32 = initialize_from %.loc12_46.5 to %.loc12_46.6 [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_46.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_46.9: <bound method> = bound_method %.loc12_45, %.loc12_46.8 [template = constants.%.34]
// CHECK:STDOUT:   %.loc12_46.10: <specific function> = specific_function %.loc12_46.9, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc12_46.2: init %i32 = call %.loc12_46.10(%.loc12_45) [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_46.11: init %i32 = converted %.loc12_45, %int.convert_checked.loc12_46.2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_46.12: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc12_46.13: init %i32 = initialize_from %.loc12_46.11 to %.loc12_46.12 [template = constants.%.36]
// CHECK:STDOUT:   %.loc12_46.14: init %.2 = struct_init (%.loc12_46.7, %.loc12_46.13) to %s.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_47: init %.2 = converted %.loc12_46.1, %.loc12_46.14 [template = constants.%struct]
// CHECK:STDOUT:   assign %s.var, %.loc12_47
// CHECK:STDOUT:   %.loc14_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_15: init type = call constants.%Int(%.loc14_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.2: type = value_of_initializer %int.make_type_signed.loc14_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.3: type = converted %int.make_type_signed.loc14_15, %.loc14_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_24: init type = call constants.%Int(%.loc14_24.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.2: type = value_of_initializer %int.make_type_signed.loc14_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.3: type = converted %int.make_type_signed.loc14_24, %.loc14_24.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_27: type = struct_type {.a: %i32, .b: %i32} [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_28: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:   %p.var: ref %.3 = var p
// CHECK:STDOUT:   %p: ref %.3 = bind_name p, %p.var
// CHECK:STDOUT:   %s.ref.loc14: ref %.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc14_32: %.3 = addr_of %s.ref.loc14
// CHECK:STDOUT:   assign %p.var, %.loc14_32
// CHECK:STDOUT:   %.loc15_10: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.2: type = converted %int.make_type_signed.loc15, %.loc15_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.3: type = ptr_type %i32 [template = constants.%.37]
// CHECK:STDOUT:   %q.var: ref %.37 = var q
// CHECK:STDOUT:   %q: ref %.37 = bind_name q, %q.var
// CHECK:STDOUT:   %s.ref.loc15: ref %.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc15_19: ref %i32 = struct_access %s.ref.loc15, element0
// CHECK:STDOUT:   %.loc15_17: %.37 = addr_of %.loc15_19
// CHECK:STDOUT:   assign %q.var, %.loc15_17
// CHECK:STDOUT:   %.loc16_10: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc16: init type = call constants.%Int(%.loc16_10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_13.1: type = value_of_initializer %int.make_type_signed.loc16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_13.2: type = converted %int.make_type_signed.loc16, %.loc16_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_13.3: type = ptr_type %i32 [template = constants.%.37]
// CHECK:STDOUT:   %r.var: ref %.37 = var r
// CHECK:STDOUT:   %r: ref %.37 = bind_name r, %r.var
// CHECK:STDOUT:   %s.ref.loc16: ref %.2 = name_ref s, %s
// CHECK:STDOUT:   %.loc16_19: ref %i32 = struct_access %s.ref.loc16, element1
// CHECK:STDOUT:   %.loc16_17: %.37 = addr_of %.loc16_19
// CHECK:STDOUT:   assign %r.var, %.loc16_17
// CHECK:STDOUT:   %.loc18_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18_11: init type = call constants.%Int(%.loc18_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_16: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc18_16: init type = call constants.%Int(%.loc18_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc18_11, %int.make_type_signed.loc18_16)
// CHECK:STDOUT:   %.loc18_19.2: type = value_of_initializer %int.make_type_signed.loc18_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.3: type = converted %int.make_type_signed.loc18_11, %.loc18_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.4: type = value_of_initializer %int.make_type_signed.loc18_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.5: type = converted %int.make_type_signed.loc18_16, %.loc18_19.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.6: type = converted %.loc18_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %t.var: ref %tuple.type.2 = var t
// CHECK:STDOUT:   %t: ref %tuple.type.2 = bind_name t, %t.var
// CHECK:STDOUT:   %.loc18_24: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18_27: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18_28.1: %tuple.type.3 = tuple_literal (%.loc18_24, %.loc18_27)
// CHECK:STDOUT:   %.loc18_28.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_28.3: <bound method> = bound_method %.loc18_24, %.loc18_28.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc18_28.4: <specific function> = specific_function %.loc18_28.3, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.1: init %i32 = call %.loc18_28.4(%.loc18_24) [template = constants.%.33]
// CHECK:STDOUT:   %.loc18_28.5: init %i32 = converted %.loc18_24, %int.convert_checked.loc18_28.1 [template = constants.%.33]
// CHECK:STDOUT:   %.loc18_28.6: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc18_28.7: init %i32 = initialize_from %.loc18_28.5 to %.loc18_28.6 [template = constants.%.33]
// CHECK:STDOUT:   %.loc18_28.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_28.9: <bound method> = bound_method %.loc18_27, %.loc18_28.8 [template = constants.%.34]
// CHECK:STDOUT:   %.loc18_28.10: <specific function> = specific_function %.loc18_28.9, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.2: init %i32 = call %.loc18_28.10(%.loc18_27) [template = constants.%.36]
// CHECK:STDOUT:   %.loc18_28.11: init %i32 = converted %.loc18_27, %int.convert_checked.loc18_28.2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc18_28.12: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc18_28.13: init %i32 = initialize_from %.loc18_28.11 to %.loc18_28.12 [template = constants.%.36]
// CHECK:STDOUT:   %.loc18_28.14: init %tuple.type.2 = tuple_init (%.loc18_28.7, %.loc18_28.13) to %t.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc18_29: init %tuple.type.2 = converted %.loc18_28.1, %.loc18_28.14 [template = constants.%tuple]
// CHECK:STDOUT:   assign %t.var, %.loc18_29
// CHECK:STDOUT:   %.loc19_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%.loc19_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.1: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.2: type = converted %int.make_type_signed.loc19, %.loc19_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.3: type = ptr_type %i32 [template = constants.%.37]
// CHECK:STDOUT:   %t0.var: ref %.37 = var t0
// CHECK:STDOUT:   %t0: ref %.37 = bind_name t0, %t0.var
// CHECK:STDOUT:   %t.ref.loc19: ref %tuple.type.2 = name_ref t, %t
// CHECK:STDOUT:   %.loc19_21: Core.IntLiteral = int_value 0 [template = constants.%.39]
// CHECK:STDOUT:   %.loc19_20: ref %i32 = tuple_access %t.ref.loc19, element0
// CHECK:STDOUT:   %.loc19_18: %.37 = addr_of %.loc19_20
// CHECK:STDOUT:   assign %t0.var, %.loc19_18
// CHECK:STDOUT:   %.loc20_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%.loc20_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_14.1: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_14.2: type = converted %int.make_type_signed.loc20, %.loc20_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_14.3: type = ptr_type %i32 [template = constants.%.37]
// CHECK:STDOUT:   %t1.var: ref %.37 = var t1
// CHECK:STDOUT:   %t1: ref %.37 = bind_name t1, %t1.var
// CHECK:STDOUT:   %t.ref.loc20: ref %tuple.type.2 = name_ref t, %t
// CHECK:STDOUT:   %.loc20_21: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc20_20: ref %i32 = tuple_access %t.ref.loc20, element1
// CHECK:STDOUT:   %.loc20_18: %.37 = addr_of %.loc20_20
// CHECK:STDOUT:   assign %t1.var, %.loc20_18
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
