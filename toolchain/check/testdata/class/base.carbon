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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base.carbon

// --- base.carbon
package Base;

base class Base {
  var b: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

// --- fail_base_after_field.carbon
package BaseAfterField;

base class Base {
}

class Derived {
  var d: i32;

  // CHECK:STDERR: fail_base_after_field.carbon:[[@LINE+4]]:3: error: `base` declaration must appear before field declarations [BaseDeclAfterFieldDecl]
  // CHECK:STDERR:   extend base: Base;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  extend base: Base;
}

// CHECK:STDOUT: --- base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ae4: <witness> = impl_witness @Base.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %ptr.11f [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b.0a3: type = struct_type {.b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.ba8: <witness> = complete_type_witness %struct_type.b.0a3 [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.69e: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.elem.344: type = unbound_element_type %Derived, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e52: <witness> = impl_witness @Derived.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: %Derived.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.d.f8f: type = struct_type {.base: %Base, .d: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.da6: <witness> = complete_type_witness %struct_type.base.d.f8f [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.b.a15: type = struct_type {.b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.a20: type = struct_type {.base: %struct_type.b.a15, .d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%int_4.940) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.208: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_7.29f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%Base.val, %int_7.0b1) [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.d07: type = tuple_type (%i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.511: type = pattern_type %tuple.type.d07 [concrete]
// CHECK:STDOUT:   %Access.type: type = fn_type @Access [concrete]
// CHECK:STDOUT:   %Access: %Access.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %return.patt: %pattern_type.fb9 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.fb9 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %return.param: ref %Derived = out_param call_param0
// CHECK:STDOUT:     %return: ref %Derived = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Access.decl: %Access.type = fn_decl @Access [concrete = constants.%Access] {
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.fb9 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.511 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.511 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc17_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc17_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17_32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc17_35.1: %tuple.type.24b = tuple_literal (%i32.loc17_27, %i32.loc17_32)
// CHECK:STDOUT:     %.loc17_35.2: type = converted %.loc17_35.1, constants.%tuple.type.d07 [concrete = constants.%tuple.type.d07]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.d07 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.d07 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Base.as.Destroy.impl: @Base.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.decl: %Base.as.Destroy.impl.Op.type = fn_decl @Base.as.Destroy.impl.Op [concrete = constants.%Base.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc3: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %ptr.11f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Base.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Base.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Derived.as.Destroy.impl: @Derived.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.decl: %Derived.as.Destroy.impl.Op.type = fn_decl @Derived.as.Destroy.impl.Op [concrete = constants.%Derived.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.605 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.605 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc7: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %ptr.404 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Derived.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Derived.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc4: %Base.elem = field_decl b, element0 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:   impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.ae4]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.b.0a3 [concrete = constants.%complete_type.ba8]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .b = %.loc4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem.69e = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc10: %Derived.elem.344 = field_decl d, element1 [concrete]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   impl_decl @Derived.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.decl), @Derived.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.e52]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.d.f8f [concrete = constants.%complete_type.da6]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .d = %.loc10
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.as.Destroy.impl.Op(%self.param: %ptr.11f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.as.Destroy.impl.Op(%self.param: %ptr.404) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make() -> %return.param: %Derived {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc14_26.1: %struct_type.b.a15 = struct_literal (%int_4)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.base.d.a20 = struct_literal (%.loc14_26.1, %int_7)
// CHECK:STDOUT:   %impl.elem0.loc14_26: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_26.1: <bound method> = bound_method %int_4, %impl.elem0.loc14_26 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc14_26: <specific function> = specific_function %impl.elem0.loc14_26, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_26.2: <bound method> = bound_method %int_4, %specific_fn.loc14_26 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_26: init %i32 = call %bound_method.loc14_26.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc14_26.2: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_26 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc14_35.2: ref %Base = class_element_access %return, element0
// CHECK:STDOUT:   %.loc14_26.3: ref %i32 = class_element_access %.loc14_35.2, element0
// CHECK:STDOUT:   %.loc14_26.4: init %i32 = initialize_from %.loc14_26.2 to %.loc14_26.3 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc14_26.5: init %Base = class_init (%.loc14_26.4), %.loc14_35.2 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc14_35.3: init %Base = converted %.loc14_26.1, %.loc14_26.5 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %impl.elem0.loc14_35: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc14_35.1: <bound method> = bound_method %int_7, %impl.elem0.loc14_35 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.208]
// CHECK:STDOUT:   %specific_fn.loc14_35: <specific function> = specific_function %impl.elem0.loc14_35, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_35.2: <bound method> = bound_method %int_7, %specific_fn.loc14_35 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_35: init %i32 = call %bound_method.loc14_35.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_35.4: init %i32 = converted %int_7, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_35 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_35.5: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc14_35.6: init %i32 = initialize_from %.loc14_35.4 to %.loc14_35.5 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc14_35.7: init %Derived = class_init (%.loc14_35.3, %.loc14_35.6), %return [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc14_36: init %Derived = converted %.loc14_35.1, %.loc14_35.7 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   return %.loc14_36 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Access(%d.param: %Derived) -> %return.param: %tuple.type.d07 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc18_11: %Derived = name_ref d, %d
// CHECK:STDOUT:   %d.ref.loc18_12: %Derived.elem.344 = name_ref d, @Derived.%.loc10 [concrete = @Derived.%.loc10]
// CHECK:STDOUT:   %.loc18_12.1: ref %i32 = class_element_access %d.ref.loc18_11, element1
// CHECK:STDOUT:   %.loc18_12.2: %i32 = bind_value %.loc18_12.1
// CHECK:STDOUT:   %d.ref.loc18_16: %Derived = name_ref d, %d
// CHECK:STDOUT:   %base.ref: %Derived.elem.69e = name_ref base, @Derived.%.loc8 [concrete = @Derived.%.loc8]
// CHECK:STDOUT:   %.loc18_17.1: ref %Base = class_element_access %d.ref.loc18_16, element0
// CHECK:STDOUT:   %.loc18_17.2: %Base = bind_value %.loc18_17.1
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc4 [concrete = @Base.%.loc4]
// CHECK:STDOUT:   %.loc18_22.1: ref %i32 = class_element_access %.loc18_17.2, element0
// CHECK:STDOUT:   %.loc18_22.2: %i32 = bind_value %.loc18_22.1
// CHECK:STDOUT:   %.loc18_24.1: %tuple.type.d07 = tuple_literal (%.loc18_12.2, %.loc18_22.2)
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return, element0
// CHECK:STDOUT:   %.loc18_24.2: init %i32 = initialize_from %.loc18_12.2 to %tuple.elem0
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return, element1
// CHECK:STDOUT:   %.loc18_24.3: init %i32 = initialize_from %.loc18_22.2 to %tuple.elem1
// CHECK:STDOUT:   %.loc18_24.4: init %tuple.type.d07 = tuple_init (%.loc18_24.2, %.loc18_24.3) to %return
// CHECK:STDOUT:   %.loc18_25: init %tuple.type.d07 = converted %.loc18_24.1, %.loc18_24.4
// CHECK:STDOUT:   return %.loc18_25 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_base_after_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ae4: <witness> = impl_witness @Base.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %ptr.11f [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e52: <witness> = impl_witness @Derived.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: %Derived.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.860: <witness> = complete_type_witness %struct_type.d [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Base.as.Destroy.impl: @Base.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.decl: %Base.as.Destroy.impl.Op.type = fn_decl @Base.as.Destroy.impl.Op [concrete = constants.%Base.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc3: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %ptr.11f = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Base.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Base.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Derived.as.Destroy.impl: @Derived.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.decl: %Derived.as.Destroy.impl.Op.type = fn_decl @Derived.as.Destroy.impl.Op [concrete = constants.%Derived.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.605 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.605 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc6: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %ptr.404 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Derived.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Derived.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:   impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.ae4]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc7: %Derived.elem = field_decl d, element0 [concrete]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:   impl_decl @Derived.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.decl), @Derived.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.e52]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.d [concrete = constants.%complete_type.860]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .d = %.loc7
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.as.Destroy.impl.Op(%self.param: %ptr.11f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.as.Destroy.impl.Op(%self.param: %ptr.404) = "no_op";
// CHECK:STDOUT:
