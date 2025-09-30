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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field_access_in_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field_access_in_value.carbon

class Class {
  var j: i32;
  var k: i32;
}

fn Test() {
  var cv: Class;
  cv.j = 1;
  cv.k = 2;
  let c: Class = cv;
  var cj: i32 = c.j;
  var ck: i32 = c.k;
}

// CHECK:STDOUT: --- field_access_in_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.j.k: type = struct_type {.j: %i32, .k: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cf7: <witness> = complete_type_witness %struct_type.j.k [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.8bd: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.d23: %type_where = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.cb3: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.d23) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.cad: %AggregateT.as_type.as.Destroy.impl.Op.type.cb3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.5a0: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.cad, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.d23) [concrete]
// CHECK:STDOUT:   %facet_value.d3d: %type_where = facet_value %Class, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.bd3: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value.d3d) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.7c2: %AggregateT.as_type.as.Destroy.impl.Op.type.bd3 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.043: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.7c2, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value.d3d) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Class.elem = field_decl j, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17: %Class.elem = field_decl k, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.j.k [concrete = constants.%complete_type.cf7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc16
// CHECK:STDOUT:   .k = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cv.patt: %pattern_type.761 = binding_pattern cv [concrete]
// CHECK:STDOUT:     %cv.var_patt: %pattern_type.761 = var_pattern %cv.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cv.var: ref %Class = var %cv.var_patt
// CHECK:STDOUT:   %Class.ref.loc21: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %cv: ref %Class = bind_name cv, %cv.var
// CHECK:STDOUT:   %cv.ref.loc22: ref %Class = name_ref cv, %cv
// CHECK:STDOUT:   %j.ref.loc22: %Class.elem = name_ref j, @Class.%.loc16 [concrete = @Class.%.loc16]
// CHECK:STDOUT:   %.loc22_5: ref %i32 = class_element_access %cv.ref.loc22, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc22_8.1: <bound method> = bound_method %int_1, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_8.2: <bound method> = bound_method %int_1, %specific_fn.loc22 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22_8.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_8: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %.loc22_5, %.loc22_8
// CHECK:STDOUT:   %cv.ref.loc23: ref %Class = name_ref cv, %cv
// CHECK:STDOUT:   %k.ref.loc23: %Class.elem = name_ref k, @Class.%.loc17 [concrete = @Class.%.loc17]
// CHECK:STDOUT:   %.loc23_5: ref %i32 = class_element_access %cv.ref.loc23, element1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc23: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc23_8.1: <bound method> = bound_method %int_2, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_8.2: <bound method> = bound_method %int_2, %specific_fn.loc23 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_8.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc23_8: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   assign %.loc23_5, %.loc23_8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.761 = binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cv.ref.loc24: ref %Class = name_ref cv, %cv
// CHECK:STDOUT:   %Class.ref.loc24: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %.loc24: %Class = bind_value %cv.ref.loc24
// CHECK:STDOUT:   %c: %Class = bind_name c, %.loc24
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %cj.patt: %pattern_type.7ce = binding_pattern cj [concrete]
// CHECK:STDOUT:     %cj.var_patt: %pattern_type.7ce = var_pattern %cj.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj.var: ref %i32 = var %cj.var_patt
// CHECK:STDOUT:   %c.ref.loc25: %Class = name_ref c, %c
// CHECK:STDOUT:   %j.ref.loc25: %Class.elem = name_ref j, @Class.%.loc16 [concrete = @Class.%.loc16]
// CHECK:STDOUT:   %.loc25_18.1: ref %i32 = class_element_access %c.ref.loc25, element0
// CHECK:STDOUT:   %.loc25_18.2: %i32 = bind_value %.loc25_18.1
// CHECK:STDOUT:   %impl.elem0.loc25: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc25_18.1: <bound method> = bound_method %.loc25_18.2, %impl.elem0.loc25
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_18.2: <bound method> = bound_method %.loc25_18.2, %specific_fn.loc25
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc25: init %i32 = call %bound_method.loc25_18.2(%.loc25_18.2)
// CHECK:STDOUT:   assign %cj.var, %Int.as.Copy.impl.Op.call.loc25
// CHECK:STDOUT:   %.loc25_11: type = splice_block %i32.loc25 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %cj: ref %i32 = bind_name cj, %cj.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ck.patt: %pattern_type.7ce = binding_pattern ck [concrete]
// CHECK:STDOUT:     %ck.var_patt: %pattern_type.7ce = var_pattern %ck.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck.var: ref %i32 = var %ck.var_patt
// CHECK:STDOUT:   %c.ref.loc26: %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc26: %Class.elem = name_ref k, @Class.%.loc17 [concrete = @Class.%.loc17]
// CHECK:STDOUT:   %.loc26_18.1: ref %i32 = class_element_access %c.ref.loc26, element1
// CHECK:STDOUT:   %.loc26_18.2: %i32 = bind_value %.loc26_18.1
// CHECK:STDOUT:   %impl.elem0.loc26: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc26_18.1: <bound method> = bound_method %.loc26_18.2, %impl.elem0.loc26
// CHECK:STDOUT:   %specific_fn.loc26: <specific function> = specific_function %impl.elem0.loc26, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc26_18.2: <bound method> = bound_method %.loc26_18.2, %specific_fn.loc26
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc26: init %i32 = call %bound_method.loc26_18.2(%.loc26_18.2)
// CHECK:STDOUT:   assign %ck.var, %Int.as.Copy.impl.Op.call.loc26
// CHECK:STDOUT:   %.loc26_11: type = splice_block %i32.loc26 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck: ref %i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %facet_value.loc26: %type_where = facet_value constants.%i32, () [concrete = constants.%facet_value.d23]
// CHECK:STDOUT:   %.loc26_3: %type_where = converted constants.%i32, %facet_value.loc26 [concrete = constants.%facet_value.d23]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc26: <bound method> = bound_method %ck.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.cad
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.cad, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.d23) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.5a0]
// CHECK:STDOUT:   %bound_method.loc26_3: <bound method> = bound_method %ck.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc26: %ptr.235 = addr_of %ck.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc26: init %empty_tuple.type = call %bound_method.loc26_3(%addr.loc26)
// CHECK:STDOUT:   %facet_value.loc25: %type_where = facet_value constants.%i32, () [concrete = constants.%facet_value.d23]
// CHECK:STDOUT:   %.loc25_3: %type_where = converted constants.%i32, %facet_value.loc25 [concrete = constants.%facet_value.d23]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc25: <bound method> = bound_method %cj.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.cad
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.cad, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.d23) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.5a0]
// CHECK:STDOUT:   %bound_method.loc25_3: <bound method> = bound_method %cj.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc25: %ptr.235 = addr_of %cj.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25_3(%addr.loc25)
// CHECK:STDOUT:   %facet_value.loc21: %type_where = facet_value constants.%Class, () [concrete = constants.%facet_value.d3d]
// CHECK:STDOUT:   %.loc21: %type_where = converted constants.%Class, %facet_value.loc21 [concrete = constants.%facet_value.d3d]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound.loc21: <bound method> = bound_method %cv.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.7c2
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.3: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.7c2, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value.d3d) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn.043]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %cv.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc21: %ptr.e71 = addr_of %cv.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call.loc21: init %empty_tuple.type = call %bound_method.loc21(%addr.loc21)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
