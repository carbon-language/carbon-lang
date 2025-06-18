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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self_conversion.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self_conversion.carbon

base class Base {
  var a: i32;
}

class Derived {
  extend base: Base;

  fn SelfBase[self: Base]() -> i32;
  fn AddrSelfBase[addr self: Base*]();
}

fn Derived.SelfBase[self: Base]() -> i32 {
  return self.a;
}

fn Derived.AddrSelfBase[addr self: Base*]() {
  (*self).a = 1;
}

fn Call(p: Derived*) -> i32 {
  (*p).AddrSelfBase();
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
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd7: <witness> = complete_type_witness %struct_type.a [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %SelfBase.type: type = fn_type @SelfBase [concrete]
// CHECK:STDOUT:   %SelfBase: %SelfBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %ptr.11f [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %AddrSelfBase.type: type = fn_type @AddrSelfBase [concrete]
// CHECK:STDOUT:   %AddrSelfBase: %AddrSelfBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [concrete]
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
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [concrete = constants.%SelfBase] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bcc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bcc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param.loc26: %Base = value_param call_param0
// CHECK:STDOUT:     %Base.ref.loc26: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc26: %Base = bind_name self, %self.param.loc26
// CHECK:STDOUT:     %return.param.loc26: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc26: ref %i32 = return_slot %return.param.loc26
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [concrete = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc30_25: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc30: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %.loc30_40: type = splice_block %ptr.loc30 [concrete = constants.%ptr.11f] {
// CHECK:STDOUT:       %Base.ref.loc30: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:       %ptr.loc30: type = ptr_type %Base.ref.loc30 [concrete = constants.%ptr.11f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc30: %ptr.11f = bind_name self, %self.param.loc30
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc34: type = splice_block %ptr [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.fd7]
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
// CHECK:STDOUT:   %SelfBase.decl: %SelfBase.type = fn_decl @SelfBase [concrete = constants.%SelfBase] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bcc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bcc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param.loc22: %Base = value_param call_param0
// CHECK:STDOUT:     %Base.ref.loc22: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:     %self.loc22: %Base = bind_name self, %self.param.loc22
// CHECK:STDOUT:     %return.param.loc22: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return.loc22: ref %i32 = return_slot %return.param.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddrSelfBase.decl: %AddrSelfBase.type = fn_decl @AddrSelfBase [concrete = constants.%AddrSelfBase] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc30_25: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc23: %ptr.11f = value_param call_param0
// CHECK:STDOUT:     %.loc23: type = splice_block %ptr.loc23 [concrete = constants.%ptr.11f] {
// CHECK:STDOUT:       %Base.ref.loc23: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:       %ptr.loc23: type = ptr_type %Base.ref.loc23 [concrete = constants.%ptr.11f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc23: %ptr.11f = bind_name self, %self.param.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete = constants.%struct_type.base.b1e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .SelfBase = %SelfBase.decl
// CHECK:STDOUT:   .AddrSelfBase = %AddrSelfBase.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SelfBase(%self.param.loc26: %Base) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Base = name_ref self, %self.loc26
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc16 [concrete = @Base.%.loc16]
// CHECK:STDOUT:   %.loc27_14.1: ref %i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc27_14.2: %i32 = bind_value %.loc27_14.1
// CHECK:STDOUT:   return %.loc27_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AddrSelfBase(%self.param.loc30: %ptr.11f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.11f = name_ref self, %self.loc30
// CHECK:STDOUT:   %.loc31_4: ref %Base = deref %self.ref
// CHECK:STDOUT:   %a.ref: %Base.elem = name_ref a, @Base.%.loc16 [concrete = @Base.%.loc16]
// CHECK:STDOUT:   %.loc31_10: ref %i32 = class_element_access %.loc31_4, element0
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc31_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc31_13.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc31_13: init %i32 = converted %int_1, %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   assign %.loc31_10, %.loc31_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param: %ptr.404) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref.loc35: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %.loc35_4.1: ref %Derived = deref %p.ref.loc35
// CHECK:STDOUT:   %AddrSelfBase.ref: %AddrSelfBase.type = name_ref AddrSelfBase, @Derived.%AddrSelfBase.decl [concrete = constants.%AddrSelfBase]
// CHECK:STDOUT:   %AddrSelfBase.bound: <bound method> = bound_method %.loc35_4.1, %AddrSelfBase.ref
// CHECK:STDOUT:   %addr.loc35_4.1: %ptr.404 = addr_of %.loc35_4.1
// CHECK:STDOUT:   %.loc35_4.2: ref %Derived = deref %addr.loc35_4.1
// CHECK:STDOUT:   %.loc35_4.3: ref %Base = class_element_access %.loc35_4.2, element0
// CHECK:STDOUT:   %addr.loc35_4.2: %ptr.11f = addr_of %.loc35_4.3
// CHECK:STDOUT:   %.loc35_4.4: %ptr.11f = converted %addr.loc35_4.1, %addr.loc35_4.2
// CHECK:STDOUT:   %AddrSelfBase.call: init %empty_tuple.type = call %AddrSelfBase.bound(%.loc35_4.4)
// CHECK:STDOUT:   %p.ref.loc36: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %.loc36_11.1: ref %Derived = deref %p.ref.loc36
// CHECK:STDOUT:   %SelfBase.ref: %SelfBase.type = name_ref SelfBase, @Derived.%SelfBase.decl [concrete = constants.%SelfBase]
// CHECK:STDOUT:   %SelfBase.bound: <bound method> = bound_method %.loc36_11.1, %SelfBase.ref
// CHECK:STDOUT:   %.loc36_11.2: ref %Base = class_element_access %.loc36_11.1, element0
// CHECK:STDOUT:   %.loc36_11.3: ref %Base = converted %.loc36_11.1, %.loc36_11.2
// CHECK:STDOUT:   %.loc36_11.4: %Base = bind_value %.loc36_11.3
// CHECK:STDOUT:   %SelfBase.call: init %i32 = call %SelfBase.bound(%.loc36_11.4)
// CHECK:STDOUT:   %.loc36_25.1: %i32 = value_of_initializer %SelfBase.call
// CHECK:STDOUT:   %.loc36_25.2: %i32 = converted %SelfBase.call, %.loc36_25.1
// CHECK:STDOUT:   return %.loc36_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
