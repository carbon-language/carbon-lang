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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_method_qualified.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_method_qualified.carbon

class Derived;

base class Base {
  fn F[self: Self]() -> i32;
  fn G[self: Derived]() -> i32;
}

class Derived {
  extend base: Base;

  fn F[self: Self]();
  fn G[self: Self]();
}

fn Call(a: Derived) -> i32 {
  return a.(Base.F)();
}

fn CallIndirect(p: Derived*) -> i32 {
  return p->(Base.F)();
}

fn PassDerivedToBase(a: Derived) -> i32 {
  return a.(Base.G)();
}

fn PassDerivedToBaseIndirect(p: Derived*) -> i32 {
  return p->(Base.G)();
}

// CHECK:STDOUT: --- base_method_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Base.G.type: type = fn_type @Base.G [concrete]
// CHECK:STDOUT:   %Base.G: %Base.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.ae4: <witness> = impl_witness @Base.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %pattern_type.1b9: type = pattern_type %ptr.11f [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.type: type = fn_type @Base.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op: %Base.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.G.type: type = fn_type @Derived.G [concrete]
// CHECK:STDOUT:   %Derived.G: %Derived.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.e52: <witness> = impl_witness @Derived.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.type: type = fn_type @Derived.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op: %Derived.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassDerivedToBase.type: type = fn_type @PassDerivedToBase [concrete]
// CHECK:STDOUT:   %PassDerivedToBase: %PassDerivedToBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.type: type = fn_type @PassDerivedToBaseIndirect [concrete]
// CHECK:STDOUT:   %PassDerivedToBaseIndirect: %PassDerivedToBaseIndirect.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Derived = %Derived.decl.loc15
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:     .PassDerivedToBase = %PassDerivedToBase.decl
// CHECK:STDOUT:     .PassDerivedToBaseIndirect = %PassDerivedToBaseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Derived.decl.loc15: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl.loc22: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %a.patt: %pattern_type.fb9 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.fb9 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc15 [concrete = constants.%Derived]
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc33: type = splice_block %ptr [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc15 [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBase.decl: %PassDerivedToBase.type = fn_decl @PassDerivedToBase [concrete = constants.%PassDerivedToBase] {
// CHECK:STDOUT:     %a.patt: %pattern_type.fb9 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.fb9 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc15 [concrete = constants.%Derived]
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.decl: %PassDerivedToBaseIndirect.type = fn_decl @PassDerivedToBaseIndirect [concrete = constants.%PassDerivedToBaseIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc41: type = splice_block %ptr [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc15 [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Base.as.Destroy.impl: constants.%Base as constants.%Destroy.type {
// CHECK:STDOUT:   %Base.as.Destroy.impl.Op.decl: %Base.as.Destroy.impl.Op.type = fn_decl @Base.as.Destroy.impl.Op [concrete = constants.%Base.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.1b9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.1b9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT: impl @Derived.as.Destroy.impl: constants.%Derived as constants.%Destroy.type {
// CHECK:STDOUT:   %Derived.as.Destroy.impl.Op.decl: %Derived.as.Destroy.impl.Op.type = fn_decl @Derived.as.Destroy.impl.Op [concrete = constants.%Derived.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.605 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.605 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc22: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
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
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc23: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.G.decl: %Derived.G.type = fn_decl @Derived.G [concrete = constants.%Derived.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Derived.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Derived.as.Destroy.impl.%Derived.as.Destroy.impl.Op.decl), @Derived.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.e52]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete = constants.%struct_type.base.b1e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc23
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .G = %Derived.G.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bcc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bcc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.G.decl: %Base.G.type = fn_decl @Base.G [concrete = constants.%Base.G] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc15 [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Base.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Base.as.Destroy.impl.%Base.as.Destroy.impl.Op.decl), @Base.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.ae4]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   .G = %Base.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.F(%self.param: %Base) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.G(%self.param: %Derived) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Base.as.Destroy.impl.Op(%self.param: %ptr.11f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.F(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.G(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Derived.as.Destroy.impl.Op(%self.param: %ptr.404) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, @Base.%Base.F.decl [concrete = constants.%Base.F]
// CHECK:STDOUT:   %Base.F.bound: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc30_10.1: ref %Base = class_element_access %a.ref, element0
// CHECK:STDOUT:   %.loc30_10.2: ref %Base = converted %a.ref, %.loc30_10.1
// CHECK:STDOUT:   %.loc30_10.3: %Base = bind_value %.loc30_10.2
// CHECK:STDOUT:   %Base.F.call: init %i32 = call %Base.F.bound(%.loc30_10.3)
// CHECK:STDOUT:   %.loc30_22.1: %i32 = value_of_initializer %Base.F.call
// CHECK:STDOUT:   %.loc30_22.2: %i32 = converted %Base.F.call, %.loc30_22.1
// CHECK:STDOUT:   return %.loc30_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect(%p.param: %ptr.404) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %F.ref: %Base.F.type = name_ref F, @Base.%Base.F.decl [concrete = constants.%Base.F]
// CHECK:STDOUT:   %.loc34_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %Base.F.bound: <bound method> = bound_method %.loc34_11.1, %F.ref
// CHECK:STDOUT:   %.loc34_11.2: ref %Base = class_element_access %.loc34_11.1, element0
// CHECK:STDOUT:   %.loc34_11.3: ref %Base = converted %.loc34_11.1, %.loc34_11.2
// CHECK:STDOUT:   %.loc34_11.4: %Base = bind_value %.loc34_11.3
// CHECK:STDOUT:   %Base.F.call: init %i32 = call %Base.F.bound(%.loc34_11.4)
// CHECK:STDOUT:   %.loc34_23.1: %i32 = value_of_initializer %Base.F.call
// CHECK:STDOUT:   %.loc34_23.2: %i32 = converted %Base.F.call, %.loc34_23.1
// CHECK:STDOUT:   return %.loc34_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBase(%a.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %G.ref: %Base.G.type = name_ref G, @Base.%Base.G.decl [concrete = constants.%Base.G]
// CHECK:STDOUT:   %Base.G.bound: <bound method> = bound_method %a.ref, %G.ref
// CHECK:STDOUT:   %Base.G.call: init %i32 = call %Base.G.bound(%a.ref)
// CHECK:STDOUT:   %.loc38_22.1: %i32 = value_of_initializer %Base.G.call
// CHECK:STDOUT:   %.loc38_22.2: %i32 = converted %Base.G.call, %.loc38_22.1
// CHECK:STDOUT:   return %.loc38_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBaseIndirect(%p.param: %ptr.404) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %G.ref: %Base.G.type = name_ref G, @Base.%Base.G.decl [concrete = constants.%Base.G]
// CHECK:STDOUT:   %.loc42_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %Base.G.bound: <bound method> = bound_method %.loc42_11.1, %G.ref
// CHECK:STDOUT:   %.loc42_11.2: %Derived = bind_value %.loc42_11.1
// CHECK:STDOUT:   %Base.G.call: init %i32 = call %Base.G.bound(%.loc42_11.2)
// CHECK:STDOUT:   %.loc42_23.1: %i32 = value_of_initializer %Base.G.call
// CHECK:STDOUT:   %.loc42_23.2: %i32 = converted %Base.G.call, %.loc42_23.1
// CHECK:STDOUT:   return %.loc42_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
