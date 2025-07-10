// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field_access.carbon

class Class {
  var j: i32;
  var k: i32;
}

fn Run() {
  var c: Class;
  c.j = 1;
  c.k = 2;
  var cj: i32 = c.j;
  var ck: i32 = c.k;
}

// CHECK:STDOUT: --- field_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.j.k: type = struct_type {.j: %i32, .k: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cf7: <witness> = complete_type_witness %struct_type.j.k [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.3, @impl.49c(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.014: <specific function> = specific_function %Op.e6a, @Op.3(%i32) [concrete]
// CHECK:STDOUT:   %Op.type.7de: type = fn_type @Op.3, @impl.49c(%Class) [concrete]
// CHECK:STDOUT:   %Op.d64: %Op.type.7de = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Op.specific_fn.618: <specific function> = specific_function %Op.d64, @Op.3(%Class) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Class.elem = field_decl j, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17: %Class.elem = field_decl k, element1 [concrete]
// CHECK:STDOUT:   %struct_type.j.k: type = struct_type {.j: %i32, .k: %i32} [concrete = constants.%struct_type.j.k]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.j.k [concrete = constants.%complete_type.cf7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc16
// CHECK:STDOUT:   .k = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.761 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Class = var %c.var_patt
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %c: ref %Class = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref.loc22: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc22: %Class.elem = name_ref j, @Class.%.loc16 [concrete = @Class.%.loc16]
// CHECK:STDOUT:   %.loc22_4: ref %i32 = class_element_access %c.ref.loc22, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc22: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc22_7.1: <bound method> = bound_method %int_1, %impl.elem0.loc22 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_7.2: <bound method> = bound_method %int_1, %specific_fn.loc22 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc22: init %i32 = call %bound_method.loc22_7.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_7: init %i32 = converted %int_1, %int.convert_checked.loc22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %.loc22_4, %.loc22_7
// CHECK:STDOUT:   %c.ref.loc23: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc23: %Class.elem = name_ref k, @Class.%.loc17 [concrete = @Class.%.loc17]
// CHECK:STDOUT:   %.loc23_4: ref %i32 = class_element_access %c.ref.loc23, element1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc23: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc23_7.1: <bound method> = bound_method %int_2, %impl.elem0.loc23 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_7.2: <bound method> = bound_method %int_2, %specific_fn.loc23 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc23: init %i32 = call %bound_method.loc23_7.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc23_7: init %i32 = converted %int_2, %int.convert_checked.loc23 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   assign %.loc23_4, %.loc23_7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cj.patt: %pattern_type.7ce = binding_pattern cj [concrete]
// CHECK:STDOUT:     %cj.var_patt: %pattern_type.7ce = var_pattern %cj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj.var: ref %i32 = var %cj.var_patt
// CHECK:STDOUT:   %c.ref.loc24: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc24: %Class.elem = name_ref j, @Class.%.loc16 [concrete = @Class.%.loc16]
// CHECK:STDOUT:   %.loc24_18.1: ref %i32 = class_element_access %c.ref.loc24, element0
// CHECK:STDOUT:   %.loc24_18.2: %i32 = bind_value %.loc24_18.1
// CHECK:STDOUT:   assign %cj.var, %.loc24_18.2
// CHECK:STDOUT:   %.loc24_11: type = splice_block %i32.loc24 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj: ref %i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ck.patt: %pattern_type.7ce = binding_pattern ck [concrete]
// CHECK:STDOUT:     %ck.var_patt: %pattern_type.7ce = var_pattern %ck.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck.var: ref %i32 = var %ck.var_patt
// CHECK:STDOUT:   %c.ref.loc25: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc25: %Class.elem = name_ref k, @Class.%.loc17 [concrete = @Class.%.loc17]
// CHECK:STDOUT:   %.loc25_18.1: ref %i32 = class_element_access %c.ref.loc25, element1
// CHECK:STDOUT:   %.loc25_18.2: %i32 = bind_value %.loc25_18.1
// CHECK:STDOUT:   assign %ck.var, %.loc25_18.2
// CHECK:STDOUT:   %.loc25_11: type = splice_block %i32.loc25 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck: ref %i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %Op.bound.loc25: <bound method> = bound_method %ck.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc25: <bound method> = bound_method %ck.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc25: %ptr.235 = addr_of %ck.var
// CHECK:STDOUT:   %no_op.loc25: init %empty_tuple.type = call %bound_method.loc25(%addr.loc25)
// CHECK:STDOUT:   %Op.bound.loc24: <bound method> = bound_method %cj.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc24: <bound method> = bound_method %cj.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc24: %ptr.235 = addr_of %cj.var
// CHECK:STDOUT:   %no_op.loc24: init %empty_tuple.type = call %bound_method.loc24(%addr.loc24)
// CHECK:STDOUT:   %Op.bound.loc21: <bound method> = bound_method %c.var, constants.%Op.d64
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.d64, @Op.3(constants.%Class) [concrete = constants.%Op.specific_fn.618]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %c.var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc21: %ptr.e71 = addr_of %c.var
// CHECK:STDOUT:   %no_op.loc21: init %empty_tuple.type = call %bound_method.loc21(%addr.loc21)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
