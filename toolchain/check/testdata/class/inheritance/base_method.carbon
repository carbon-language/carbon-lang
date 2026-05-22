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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/base_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/base_method.carbon

base class Base {
  var a: i32;

  fn F[ref self: Self]();
}

fn Base.F[ref self: Self]() {
  self.a = 1;
}

class Derived {
  extend base: Base;
}

fn Call(p: Derived*) {
  (*p).F();
}

// CHECK:STDOUT: --- base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.85f: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd3: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %struct_type.base.e63: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.4e8: <witness> = complete_type_witness %struct_type.base.e63 [concrete]
// CHECK:STDOUT:   %ptr.128: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.66e: type = pattern_type %ptr.128 [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.85f = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.85f = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc21: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc21: ref %Base = ref_binding self, %self.param.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.66e = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.66e = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %ptr.128 = value_param call_param0
// CHECK:STDOUT:     %.loc29: type = splice_block %ptr [concrete = constants.%ptr.128] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.128]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.128 = value_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc16: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.85f = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.85f = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc18: ref %Base = ref_param call_param0
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc18: ref %Base = ref_binding self, %self.param.loc18
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a [concrete = constants.%complete_type.fd3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc26: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e63 [concrete = constants.%complete_type.4e8]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc26
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F(%self.param.loc21: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: ref %Base = name_ref self, %self.loc21
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc16 [concrete = @Base.%.loc16]
// CHECK:STDOUT:   %.loc22_7: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc22_10.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_10.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc22_10.2(%int_1) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc22_10: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   assign %.loc22_7, %.loc22_10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param: %ptr.128) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.128 = name_ref p, %p
// CHECK:STDOUT:   %.loc30_4.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, @Base.%Base.F.decl [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.bound: <bound method> = bound_method %.loc30_4.1, %F.ref
// CHECK:STDOUT:   %.loc30_4.2: ref %Base = class_element_access %.loc30_4.1, element0
// CHECK:STDOUT:   %.loc30_4.3: ref %Base = converted %.loc30_4.1, %.loc30_4.2
// CHECK:STDOUT:   %Base.F.call: init %empty_tuple.type = call %Base.F.bound(%.loc30_4.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
