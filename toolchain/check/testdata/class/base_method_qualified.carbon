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
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.10: type = ptr_type %Derived [template]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [template]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [template]
// CHECK:STDOUT:   %PassDerivedToBase.type: type = fn_type @PassDerivedToBase [template]
// CHECK:STDOUT:   %PassDerivedToBase: %PassDerivedToBase.type = struct_value () [template]
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.type: type = fn_type @PassDerivedToBaseIndirect [template]
// CHECK:STDOUT:   %PassDerivedToBaseIndirect: %PassDerivedToBaseIndirect.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Derived = %Derived.decl.loc11
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:     .CallIndirect = %CallIndirect.decl
// CHECK:STDOUT:     .PassDerivedToBase = %PassDerivedToBase.decl
// CHECK:STDOUT:     .PassDerivedToBaseIndirect = %PassDerivedToBaseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Derived.decl.loc11: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl.loc18: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %a.patt: %Derived = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Derived = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc25_24.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc25_24.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc25_24.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc25_24.3: type = converted %int.make_type_signed, %.loc25_24.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [template = constants.%CallIndirect] {
// CHECK:STDOUT:     %p.patt: %.10 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.10 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc29_27: type = ptr_type %Derived [template = constants.%.10]
// CHECK:STDOUT:     %.loc29_33.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc29_33.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_33.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc29_33.3: type = converted %int.make_type_signed, %.loc29_33.2 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %.10 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.10 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBase.decl: %PassDerivedToBase.type = fn_decl @PassDerivedToBase [template = constants.%PassDerivedToBase] {
// CHECK:STDOUT:     %a.patt: %Derived = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %Derived = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc33_37.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc33_37.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc33_37.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc33_37.3: type = converted %int.make_type_signed, %.loc33_37.2 [template = constants.%i32]
// CHECK:STDOUT:     %a.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.decl: %PassDerivedToBaseIndirect.type = fn_decl @PassDerivedToBaseIndirect [template = constants.%PassDerivedToBaseIndirect] {
// CHECK:STDOUT:     %p.patt: %.10 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %.10 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc37_40: type = ptr_type %Derived [template = constants.%.10]
// CHECK:STDOUT:     %.loc37_46.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc37_46.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc37_46.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc37_46.3: type = converted %int.make_type_signed, %.loc37_46.2 [template = constants.%i32]
// CHECK:STDOUT:     %p.param: %.10 = value_param runtime_param0
// CHECK:STDOUT:     %p: %.10 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc19: %.5 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.2 = fn_decl @G.2 [template = constants.%G.2] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc19
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %.loc14_25.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc14_25.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_25.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc14_25.3: type = converted %int.make_type_signed, %.loc14_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %Base = value_param runtime_param0
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.1 = fn_decl @G.1 [template = constants.%G.1] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc15_28.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc15_28.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_28.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc15_28.3: type = converted %int.make_type_signed, %.loc15_28.2 [template = constants.%i32]
// CHECK:STDOUT:     %self.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
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
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @Base.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc26_11: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc26_10.1: ref %Base = class_element_access %a.ref, element0
// CHECK:STDOUT:   %.loc26_10.2: ref %Base = converted %a.ref, %.loc26_10.1
// CHECK:STDOUT:   %.loc26_10.3: %Base = bind_value %.loc26_10.2
// CHECK:STDOUT:   %F.call: init %i32 = call %.loc26_11(%.loc26_10.3)
// CHECK:STDOUT:   %.loc26_22.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc26_22.2: %i32 = converted %F.call, %.loc26_22.1
// CHECK:STDOUT:   return %.loc26_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect(%p.param_patt: %.10) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.10 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @Base.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc30_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc30_11.2: <bound method> = bound_method %.loc30_11.1, %F.ref
// CHECK:STDOUT:   %.loc30_11.3: ref %Base = class_element_access %.loc30_11.1, element0
// CHECK:STDOUT:   %.loc30_11.4: ref %Base = converted %.loc30_11.1, %.loc30_11.3
// CHECK:STDOUT:   %.loc30_11.5: %Base = bind_value %.loc30_11.4
// CHECK:STDOUT:   %F.call: init %i32 = call %.loc30_11.2(%.loc30_11.5)
// CHECK:STDOUT:   %.loc30_23.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc30_23.2: %i32 = converted %F.call, %.loc30_23.1
// CHECK:STDOUT:   return %.loc30_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBase(%a.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.1 = name_ref G, @Base.%G.decl [template = constants.%G.1]
// CHECK:STDOUT:   %.loc34_11: <bound method> = bound_method %a.ref, %G.ref
// CHECK:STDOUT:   %G.call: init %i32 = call %.loc34_11(%a.ref)
// CHECK:STDOUT:   %.loc34_22.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc34_22.2: %i32 = converted %G.call, %.loc34_22.1
// CHECK:STDOUT:   return %.loc34_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBaseIndirect(%p.param_patt: %.10) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.10 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.1 = name_ref G, @Base.%G.decl [template = constants.%G.1]
// CHECK:STDOUT:   %.loc38_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc38_11.2: <bound method> = bound_method %.loc38_11.1, %G.ref
// CHECK:STDOUT:   %.loc38_11.3: %Derived = bind_value %.loc38_11.1
// CHECK:STDOUT:   %G.call: init %i32 = call %.loc38_11.2(%.loc38_11.3)
// CHECK:STDOUT:   %.loc38_23.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc38_23.2: %i32 = converted %G.call, %.loc38_23.1
// CHECK:STDOUT:   return %.loc38_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
