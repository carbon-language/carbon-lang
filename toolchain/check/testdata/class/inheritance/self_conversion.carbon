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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/self_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/self_conversion.carbon

base class Base {
  var a: i32;
}

class Derived {
  extend base: Base;

  fn SelfBase[self: Base]() -> i32;
  fn RefSelfBase[ref self: Base]();
}

fn Derived.SelfBase[self: Base]() -> i32 {
  return self.a;
}

fn Derived.RefSelfBase[ref self: Base]() {
  self.a = 1;
}

fn Call(p: Derived*) -> i32 {
  (*p).RefSelfBase();
  return (*p).SelfBase();
}

// CHECK:STDOUT: --- self_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd7: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.101: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Derived.SelfBase.type: type = fn_type @Derived.SelfBase [concrete]
// CHECK:STDOUT:   %Derived.SelfBase: %Derived.SelfBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.RefSelfBase.type: type = fn_type @Derived.RefSelfBase [concrete]
// CHECK:STDOUT:   %Derived.RefSelfBase: %Derived.RefSelfBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.27a: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.5a1: <witness> = complete_type_witness %struct_type.base.27a [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.30d: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.aa9: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.44b: %Int.as.Copy.impl.Op.type.aa9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.cee: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.348: type = fn_type_with_self_type %Copy.WithSelf.Op.type.cee, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.44b, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ac5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ac5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b91, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.f74: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.0dd: type = pattern_type %ptr.f74 [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Derived.SelfBase.decl: %Derived.SelfBase.type = fn_decl @Derived.SelfBase [concrete = constants.%Derived.SelfBase] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc26: Core.Form = init_form %i32.loc26 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %self.param.loc26: %Base = value_param call_param0
// CHECK:STDOUT:     %Base.ref.loc26: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc26: %Base = value_binding self, %self.param.loc26
// CHECK:STDOUT:     %return.param.loc26: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc26: ref %i32 = return_slot %return.param.loc26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.RefSelfBase.decl: %Derived.RefSelfBase.type = fn_decl @Derived.RefSelfBase [concrete = constants.%Derived.RefSelfBase] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc30: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %Base.ref.loc30: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc30: ref %Base = ref_binding self, %self.param.loc30
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.0dd = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.0dd = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc34_25: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %p.param: %ptr.f74 = value_param call_param0
// CHECK:STDOUT:     %.loc34_19: type = splice_block %ptr [concrete = constants.%ptr.f74] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.f74]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.f74 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc20: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %Derived.SelfBase.decl: %Derived.SelfBase.type = fn_decl @Derived.SelfBase [concrete = constants.%Derived.SelfBase] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22: Core.Form = init_form %i32.loc22 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %self.param.loc22: %Base = value_param call_param0
// CHECK:STDOUT:     %Base.ref.loc22: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc22: %Base = value_binding self, %self.param.loc22
// CHECK:STDOUT:     %return.param.loc22: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc22: ref %i32 = return_slot %return.param.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.RefSelfBase.decl: %Derived.RefSelfBase.type = fn_decl @Derived.RefSelfBase [concrete = constants.%Derived.RefSelfBase] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc23: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %Base.ref.loc23: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc23: ref %Base = ref_binding self, %self.param.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.27a [concrete = constants.%complete_type.5a1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .SelfBase = %Derived.SelfBase.decl
// CHECK:STDOUT:   .RefSelfBase = %Derived.RefSelfBase.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.SelfBase(%self.param.loc26: %Base) -> out %return.param.loc26: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Base = name_ref self, %self.loc26
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc16 [concrete = @Base.%.loc16]
// CHECK:STDOUT:   %.loc27_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc27_14.2: %i32 = acquire_value %.loc27_14.1
// CHECK:STDOUT:   %impl.elem0: %.348 = impl_witness_access constants.%Copy.impl_witness.30d, element0 [concrete = constants.%Int.as.Copy.impl.Op.44b]
// CHECK:STDOUT:   %bound_method.loc27_14.1: <bound method> = bound_method %.loc27_14.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_14.2: <bound method> = bound_method %.loc27_14.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc27_14.2(%.loc27_14.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.RefSelfBase(%self.param.loc30: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: ref %Base = name_ref self, %self.loc30
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc16 [concrete = @Base.%.loc16]
// CHECK:STDOUT:   %.loc31_7: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc31_10.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_10.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc31_10.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc31_10: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %.loc31_7, %.loc31_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param: %ptr.f74) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc35: %ptr.f74 = name_ref p, %p
// CHECK:STDOUT:   %.loc35_4.1: ref %Derived = deref %p.ref.loc35
// CHECK:STDOUT:   %RefSelfBase.ref: %Derived.RefSelfBase.type = name_ref RefSelfBase, @Derived.%Derived.RefSelfBase.decl [concrete = constants.%Derived.RefSelfBase]
// CHECK:STDOUT:   %Derived.RefSelfBase.bound: <bound method> = bound_method %.loc35_4.1, %RefSelfBase.ref
// CHECK:STDOUT:   %.loc35_4.2: ref %Base = class_element_access %.loc35_4.1, element0
// CHECK:STDOUT:   %.loc35_4.3: ref %Base = converted %.loc35_4.1, %.loc35_4.2
// CHECK:STDOUT:   %Derived.RefSelfBase.call: init %empty_tuple.type = call %Derived.RefSelfBase.bound(%.loc35_4.3)
// CHECK:STDOUT:   %p.ref.loc36: %ptr.f74 = name_ref p, %p
// CHECK:STDOUT:   %.loc36_11.1: ref %Derived = deref %p.ref.loc36
// CHECK:STDOUT:   %SelfBase.ref: %Derived.SelfBase.type = name_ref SelfBase, @Derived.%Derived.SelfBase.decl [concrete = constants.%Derived.SelfBase]
// CHECK:STDOUT:   %Derived.SelfBase.bound: <bound method> = bound_method %.loc36_11.1, %SelfBase.ref
// CHECK:STDOUT:   %.loc36_11.2: ref %Base = class_element_access %.loc36_11.1, element0
// CHECK:STDOUT:   %.loc36_11.3: ref %Base = converted %.loc36_11.1, %.loc36_11.2
// CHECK:STDOUT:   %.loc36_11.4: %Base = acquire_value %.loc36_11.3
// CHECK:STDOUT:   %Derived.SelfBase.call: init %i32 = call %Derived.SelfBase.bound(%.loc36_11.4)
// CHECK:STDOUT:   return %Derived.SelfBase.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
