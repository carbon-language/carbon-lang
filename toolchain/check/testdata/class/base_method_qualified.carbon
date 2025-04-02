// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.6ee: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.663: %G.type.6ee = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.04c: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.07e: %G.type.04c = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.b1e: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base.b1e [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
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
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Derived = %Derived.decl.loc11
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:     .PassDerivedToBase = %PassDerivedToBase.decl
// CHECK:STDOUT:     .PassDerivedToBaseIndirect = %PassDerivedToBaseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Derived.decl.loc11: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl.loc18: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %a.patt: %Derived = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Derived = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [concrete = constants.%Derived]
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr.404 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.404 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc29: type = splice_block %ptr [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBase.decl: %PassDerivedToBase.type = fn_decl @PassDerivedToBase [concrete = constants.%PassDerivedToBase] {
// CHECK:STDOUT:     %a.patt: %Derived = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Derived = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [concrete = constants.%Derived]
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.decl: %PassDerivedToBaseIndirect.type = fn_decl @PassDerivedToBaseIndirect [concrete = constants.%PassDerivedToBaseIndirect] {
// CHECK:STDOUT:     %p.patt: %ptr.404 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.404 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc37: type = splice_block %ptr [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc19: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.04c = fn_decl @G.2 [concrete = constants.%G.07e] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b1e [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc19
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.1 [concrete = constants.%F.d17] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.6ee = fn_decl @G.1 [concrete = constants.%G.663] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Derived = <poisoned>
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[%self.param_patt: %Base]() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1[%self.param_patt: %Derived]() -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[%self.param_patt: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.7c6 = name_ref F, @Base.%F.decl [concrete = constants.%F.d17]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc26_10.1: ref %Base = class_element_access %a.ref, element0
// CHECK:STDOUT:   %.loc26_10.2: ref %Base = converted %a.ref, %.loc26_10.1
// CHECK:STDOUT:   %.loc26_10.3: %Base = bind_value %.loc26_10.2
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%.loc26_10.3)
// CHECK:STDOUT:   %.loc26_22.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc26_22.2: %i32 = converted %F.call, %.loc26_22.1
// CHECK:STDOUT:   return %.loc26_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect(%p.param_patt: %ptr.404) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.7c6 = name_ref F, @Base.%F.decl [concrete = constants.%F.d17]
// CHECK:STDOUT:   %.loc30_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc30_11.1, %F.ref
// CHECK:STDOUT:   %.loc30_11.2: ref %Base = class_element_access %.loc30_11.1, element0
// CHECK:STDOUT:   %.loc30_11.3: ref %Base = converted %.loc30_11.1, %.loc30_11.2
// CHECK:STDOUT:   %.loc30_11.4: %Base = bind_value %.loc30_11.3
// CHECK:STDOUT:   %F.call: init %i32 = call %F.bound(%.loc30_11.4)
// CHECK:STDOUT:   %.loc30_23.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc30_23.2: %i32 = converted %F.call, %.loc30_23.1
// CHECK:STDOUT:   return %.loc30_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBase(%a.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.6ee = name_ref G, @Base.%G.decl [concrete = constants.%G.663]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %a.ref, %G.ref
// CHECK:STDOUT:   %G.call: init %i32 = call %G.bound(%a.ref)
// CHECK:STDOUT:   %.loc34_22.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc34_22.2: %i32 = converted %G.call, %.loc34_22.1
// CHECK:STDOUT:   return %.loc34_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBaseIndirect(%p.param_patt: %ptr.404) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.6ee = name_ref G, @Base.%G.decl [concrete = constants.%G.663]
// CHECK:STDOUT:   %.loc38_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %.loc38_11.1, %G.ref
// CHECK:STDOUT:   %.loc38_11.2: %Derived = bind_value %.loc38_11.1
// CHECK:STDOUT:   %G.call: init %i32 = call %G.bound(%.loc38_11.2)
// CHECK:STDOUT:   %.loc38_23.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc38_23.2: %i32 = converted %G.call, %.loc38_23.1
// CHECK:STDOUT:   return %.loc38_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
