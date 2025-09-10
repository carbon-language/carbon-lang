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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.73b: <witness> = impl_witness @Class.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: %Class.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.j.k: type = struct_type {.j: %i32, .k: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cf7: <witness> = complete_type_witness %struct_type.j.k [concrete]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b59: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %bound_method.f6f: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f0b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.af5: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.87e: %Int.as.Copy.impl.Op.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.26d: %Copy.type = facet_value %i32, (%Copy.impl_witness.f0b) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.26d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.87e, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.11b: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.054: %Int.as.Destroy.impl.Op.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Destroy.impl.Op.054, @Int.as.Destroy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT: impl @Class.as.Destroy.impl: @Class.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.decl: %Class.as.Destroy.impl.Op.type = fn_decl @Class.as.Destroy.impl.Op [concrete = constants.%Class.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.796 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.796 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e71 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:     %self: %ptr.e71 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Class.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Class.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32.loc16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Class.elem = field_decl j, element0 [concrete]
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17: %Class.elem = field_decl k, element1 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [concrete = constants.%Class]
// CHECK:STDOUT:   impl_decl @Class.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Class.as.Destroy.impl.%Class.as.Destroy.impl.Op.decl), @Class.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.73b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.j.k [concrete = constants.%complete_type.cf7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc16
// CHECK:STDOUT:   .k = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Class.as.Destroy.impl.Op(%self.param: %ptr.e71) = "no_op";
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
// CHECK:STDOUT:   %impl.elem0.loc22: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc22_7.1: <bound method> = bound_method %int_1, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a02]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_7.2: <bound method> = bound_method %int_1, %specific_fn.loc22 [concrete = constants.%bound_method.b59]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22_7.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc22_7: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %.loc22_4, %.loc22_7
// CHECK:STDOUT:   %c.ref.loc23: ref %Class = name_ref c, %c
// CHECK:STDOUT:   %k.ref.loc23: %Class.elem = name_ref k, @Class.%.loc17 [concrete = @Class.%.loc17]
// CHECK:STDOUT:   %.loc23_4: ref %i32 = class_element_access %c.ref.loc23, element1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc23: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc23_7.1: <bound method> = bound_method %int_2, %impl.elem0.loc23 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.2d9]
// CHECK:STDOUT:   %specific_fn.loc23: <specific function> = specific_function %impl.elem0.loc23, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_7.2: <bound method> = bound_method %int_2, %specific_fn.loc23 [concrete = constants.%bound_method.f6f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23: init %i32 = call %bound_method.loc23_7.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc23_7: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc23 [concrete = constants.%int_2.ef8]
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
// CHECK:STDOUT:   %impl.elem0.loc24: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc24_18.1: <bound method> = bound_method %.loc24_18.2, %impl.elem0.loc24
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_18.2: <bound method> = bound_method %.loc24_18.2, %specific_fn.loc24
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc24: init %i32 = call %bound_method.loc24_18.2(%.loc24_18.2)
// CHECK:STDOUT:   assign %cj.var, %Int.as.Copy.impl.Op.call.loc24
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
// CHECK:STDOUT:   %impl.elem0.loc25: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc25_18.1: <bound method> = bound_method %.loc25_18.2, %impl.elem0.loc25
// CHECK:STDOUT:   %specific_fn.loc25: <specific function> = specific_function %impl.elem0.loc25, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_18.2: <bound method> = bound_method %.loc25_18.2, %specific_fn.loc25
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc25: init %i32 = call %bound_method.loc25_18.2(%.loc25_18.2)
// CHECK:STDOUT:   assign %ck.var, %Int.as.Copy.impl.Op.call.loc25
// CHECK:STDOUT:   %.loc25_11: type = splice_block %i32.loc25 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc25: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc25: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ck: ref %i32 = bind_name ck, %ck.var
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound.loc25: <bound method> = bound_method %ck.var, constants.%Int.as.Destroy.impl.Op.054
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%Int.as.Destroy.impl.Op.054, @Int.as.Destroy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_3: <bound method> = bound_method %ck.var, %Int.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc25: %ptr.235 = addr_of %ck.var
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call.loc25: init %empty_tuple.type = call %bound_method.loc25_3(%addr.loc25)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound.loc24: <bound method> = bound_method %cj.var, constants.%Int.as.Destroy.impl.Op.054
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%Int.as.Destroy.impl.Op.054, @Int.as.Destroy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_3: <bound method> = bound_method %cj.var, %Int.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc24: %ptr.235 = addr_of %cj.var
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call.loc24: init %empty_tuple.type = call %bound_method.loc24_3(%addr.loc24)
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%Class.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc21: %ptr.e71 = addr_of %c.var
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Class.as.Destroy.impl.Op.bound(%addr.loc21)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
