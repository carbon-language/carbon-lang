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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %struct_type.a.b.1 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:   %int_32.loc12_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_15: init type = call constants.%Int(%int_32.loc12_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.1: type = value_of_initializer %int.make_type_signed.loc12_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_15.2: type = converted %int.make_type_signed.loc12_15, %.loc12_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc12_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12_24: init type = call constants.%Int(%int_32.loc12_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.1: type = value_of_initializer %int.make_type_signed.loc12_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.2: type = converted %int.make_type_signed.loc12_24, %.loc12_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b.loc12: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %s.var: ref %struct_type.a.b.1 = var s
// CHECK:STDOUT:   %s: ref %struct_type.a.b.1 = bind_name s, %s.var
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc12_46.1: %struct_type.a.b.2 = struct_literal (%int_1.loc12, %int_2.loc12)
// CHECK:STDOUT:   %impl.elem0.loc12_46.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc12_46.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_46.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_46.1: <specific function> = specific_function %Convert.bound.loc12_46.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12_46.1: init %i32 = call %Convert.specific_fn.loc12_46.1(%int_1.loc12) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_46.2: init %i32 = converted %int_1.loc12, %int.convert_checked.loc12_46.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc12_46.3: ref %i32 = struct_access %s.var, element0
// CHECK:STDOUT:   %.loc12_46.4: init %i32 = initialize_from %.loc12_46.2 to %.loc12_46.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc12_46.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc12_46.2: <bound method> = bound_method %int_2.loc12, %impl.elem0.loc12_46.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc12_46.2: <specific function> = specific_function %Convert.bound.loc12_46.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc12_46.2: init %i32 = call %Convert.specific_fn.loc12_46.2(%int_2.loc12) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_46.5: init %i32 = converted %int_2.loc12, %int.convert_checked.loc12_46.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_46.6: ref %i32 = struct_access %s.var, element1
// CHECK:STDOUT:   %.loc12_46.7: init %i32 = initialize_from %.loc12_46.5 to %.loc12_46.6 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc12_46.8: init %struct_type.a.b.1 = struct_init (%.loc12_46.4, %.loc12_46.7) to %s.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc12_47: init %struct_type.a.b.1 = converted %.loc12_46.1, %.loc12_46.8 [template = constants.%struct]
// CHECK:STDOUT:   assign %s.var, %.loc12_47
// CHECK:STDOUT:   %int_32.loc14_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_15: init type = call constants.%Int(%int_32.loc14_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.1: type = value_of_initializer %int.make_type_signed.loc14_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_15.2: type = converted %int.make_type_signed.loc14_15, %.loc14_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_24: init type = call constants.%Int(%int_32.loc14_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.1: type = value_of_initializer %int.make_type_signed.loc14_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_24.2: type = converted %int.make_type_signed.loc14_24, %.loc14_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b.loc14: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %ptr.loc14: type = ptr_type %struct_type.a.b.1 [template = constants.%ptr.1]
// CHECK:STDOUT:   %p.var: ref %ptr.1 = var p
// CHECK:STDOUT:   %p: ref %ptr.1 = bind_name p, %p.var
// CHECK:STDOUT:   %s.ref.loc14: ref %struct_type.a.b.1 = name_ref s, %s
// CHECK:STDOUT:   %addr.loc14: %ptr.1 = addr_of %s.ref.loc14
// CHECK:STDOUT:   assign %p.var, %addr.loc14
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_13.2: type = converted %int.make_type_signed.loc15, %.loc15_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc15: type = ptr_type %i32 [template = constants.%ptr.2]
// CHECK:STDOUT:   %q.var: ref %ptr.2 = var q
// CHECK:STDOUT:   %q: ref %ptr.2 = bind_name q, %q.var
// CHECK:STDOUT:   %s.ref.loc15: ref %struct_type.a.b.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc15_19: ref %i32 = struct_access %s.ref.loc15, element0
// CHECK:STDOUT:   %addr.loc15: %ptr.2 = addr_of %.loc15_19
// CHECK:STDOUT:   assign %q.var, %addr.loc15
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16: init type = call constants.%Int(%int_32.loc16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_13.1: type = value_of_initializer %int.make_type_signed.loc16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_13.2: type = converted %int.make_type_signed.loc16, %.loc16_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc16: type = ptr_type %i32 [template = constants.%ptr.2]
// CHECK:STDOUT:   %r.var: ref %ptr.2 = var r
// CHECK:STDOUT:   %r: ref %ptr.2 = bind_name r, %r.var
// CHECK:STDOUT:   %s.ref.loc16: ref %struct_type.a.b.1 = name_ref s, %s
// CHECK:STDOUT:   %.loc16_19: ref %i32 = struct_access %s.ref.loc16, element1
// CHECK:STDOUT:   %addr.loc16: %ptr.2 = addr_of %.loc16_19
// CHECK:STDOUT:   assign %r.var, %addr.loc16
// CHECK:STDOUT:   %int_32.loc18_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc18_11: init type = call constants.%Int(%int_32.loc18_11) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc18_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc18_16: init type = call constants.%Int(%int_32.loc18_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc18_11, %int.make_type_signed.loc18_16)
// CHECK:STDOUT:   %.loc18_19.2: type = value_of_initializer %int.make_type_signed.loc18_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.3: type = converted %int.make_type_signed.loc18_11, %.loc18_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.4: type = value_of_initializer %int.make_type_signed.loc18_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.5: type = converted %int.make_type_signed.loc18_16, %.loc18_19.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc18_19.6: type = converted %.loc18_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %t.var: ref %tuple.type.2 = var t
// CHECK:STDOUT:   %t: ref %tuple.type.2 = bind_name t, %t.var
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc18: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc18_28.1: %tuple.type.3 = tuple_literal (%int_1.loc18, %int_2.loc18)
// CHECK:STDOUT:   %impl.elem0.loc18_28.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc18_28.1: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18_28.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_28.1: <specific function> = specific_function %Convert.bound.loc18_28.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.1: init %i32 = call %Convert.specific_fn.loc18_28.1(%int_1.loc18) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc18_28.2: init %i32 = converted %int_1.loc18, %int.convert_checked.loc18_28.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple.elem0.loc18: ref %i32 = tuple_access %t.var, element0
// CHECK:STDOUT:   %.loc18_28.3: init %i32 = initialize_from %.loc18_28.2 to %tuple.elem0.loc18 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc18_28.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc18_28.2: <bound method> = bound_method %int_2.loc18, %impl.elem0.loc18_28.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc18_28.2: <specific function> = specific_function %Convert.bound.loc18_28.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc18_28.2: init %i32 = call %Convert.specific_fn.loc18_28.2(%int_2.loc18) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc18_28.4: init %i32 = converted %int_2.loc18, %int.convert_checked.loc18_28.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.elem1.loc18: ref %i32 = tuple_access %t.var, element1
// CHECK:STDOUT:   %.loc18_28.5: init %i32 = initialize_from %.loc18_28.4 to %tuple.elem1.loc18 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc18_28.6: init %tuple.type.2 = tuple_init (%.loc18_28.3, %.loc18_28.5) to %t.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc18_29: init %tuple.type.2 = converted %.loc18_28.1, %.loc18_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %t.var, %.loc18_29
// CHECK:STDOUT:   %int_32.loc19: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19: init type = call constants.%Int(%int_32.loc19) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.1: type = value_of_initializer %int.make_type_signed.loc19 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_14.2: type = converted %int.make_type_signed.loc19, %.loc19_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc19: type = ptr_type %i32 [template = constants.%ptr.2]
// CHECK:STDOUT:   %t0.var: ref %ptr.2 = var t0
// CHECK:STDOUT:   %t0: ref %ptr.2 = bind_name t0, %t0.var
// CHECK:STDOUT:   %t.ref.loc19: ref %tuple.type.2 = name_ref t, %t
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc19: ref %i32 = tuple_access %t.ref.loc19, element0
// CHECK:STDOUT:   %addr.loc19: %ptr.2 = addr_of %tuple.elem0.loc19
// CHECK:STDOUT:   assign %t0.var, %addr.loc19
// CHECK:STDOUT:   %int_32.loc20: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc20: init type = call constants.%Int(%int_32.loc20) [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_14.1: type = value_of_initializer %int.make_type_signed.loc20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc20_14.2: type = converted %int.make_type_signed.loc20, %.loc20_14.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr.loc20: type = ptr_type %i32 [template = constants.%ptr.2]
// CHECK:STDOUT:   %t1.var: ref %ptr.2 = var t1
// CHECK:STDOUT:   %t1: ref %ptr.2 = bind_name t1, %t1.var
// CHECK:STDOUT:   %t.ref.loc20: ref %tuple.type.2 = name_ref t, %t
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %tuple.elem1.loc20: ref %i32 = tuple_access %t.ref.loc20, element1
// CHECK:STDOUT:   %addr.loc20: %ptr.2 = addr_of %tuple.elem1.loc20
// CHECK:STDOUT:   assign %t1.var, %addr.loc20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
