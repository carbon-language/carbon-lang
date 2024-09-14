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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %.3} [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %.8: type = ptr_type %Derived [template]
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
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %Derived.decl.loc11: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Derived.decl.loc18: type = class_decl @Derived [template = constants.%Derived] {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %Derived.ref.loc25: type = name_ref Derived, %Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %a.loc25_9.1: %Derived = param a, runtime_param0
// CHECK:STDOUT:     @Call.%a: %Derived = bind_name a, %a.loc25_9.1
// CHECK:STDOUT:     %int.make_type_32.loc25: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc25_24.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:     %.loc25_24.2: type = converted %int.make_type_32.loc25, %.loc25_24.1 [template = i32]
// CHECK:STDOUT:     @Call.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [template = constants.%CallIndirect] {
// CHECK:STDOUT:     %Derived.ref.loc29: type = name_ref Derived, %Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc29_27: type = ptr_type %Derived [template = constants.%.8]
// CHECK:STDOUT:     %p.loc29_17.1: %.8 = param p, runtime_param0
// CHECK:STDOUT:     @CallIndirect.%p: %.8 = bind_name p, %p.loc29_17.1
// CHECK:STDOUT:     %int.make_type_32.loc29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc29_33.1: type = value_of_initializer %int.make_type_32.loc29 [template = i32]
// CHECK:STDOUT:     %.loc29_33.2: type = converted %int.make_type_32.loc29, %.loc29_33.1 [template = i32]
// CHECK:STDOUT:     @CallIndirect.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBase.decl: %PassDerivedToBase.type = fn_decl @PassDerivedToBase [template = constants.%PassDerivedToBase] {
// CHECK:STDOUT:     %Derived.ref.loc33: type = name_ref Derived, %Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %a.loc33_22.1: %Derived = param a, runtime_param0
// CHECK:STDOUT:     @PassDerivedToBase.%a: %Derived = bind_name a, %a.loc33_22.1
// CHECK:STDOUT:     %int.make_type_32.loc33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc33_37.1: type = value_of_initializer %int.make_type_32.loc33 [template = i32]
// CHECK:STDOUT:     %.loc33_37.2: type = converted %int.make_type_32.loc33, %.loc33_37.1 [template = i32]
// CHECK:STDOUT:     @PassDerivedToBase.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.decl: %PassDerivedToBaseIndirect.type = fn_decl @PassDerivedToBaseIndirect [template = constants.%PassDerivedToBaseIndirect] {
// CHECK:STDOUT:     %Derived.ref.loc37: type = name_ref Derived, %Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %.loc37_40: type = ptr_type %Derived [template = constants.%.8]
// CHECK:STDOUT:     %p.loc37_30.1: %.8 = param p, runtime_param0
// CHECK:STDOUT:     @PassDerivedToBaseIndirect.%p: %.8 = bind_name p, %p.loc37_30.1
// CHECK:STDOUT:     %int.make_type_32.loc37: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc37_46.1: type = value_of_initializer %int.make_type_32.loc37 [template = i32]
// CHECK:STDOUT:     %.loc37_46.2: type = converted %int.make_type_32.loc37, %.loc37_46.1 [template = i32]
// CHECK:STDOUT:     @PassDerivedToBaseIndirect.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc19: %.4 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc21_8.1: %Derived = param self, runtime_param0
// CHECK:STDOUT:     %self.loc21_8.2: %Derived = bind_name self, %self.loc21_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.2 = fn_decl @G.2 [template = constants.%G.2] {
// CHECK:STDOUT:     %Self.ref.loc22: type = name_ref Self, constants.%Derived [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc22_8.1: %Derived = param self, runtime_param0
// CHECK:STDOUT:     %self.loc22_8.2: %Derived = bind_name self, %self.loc22_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc19
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [template = constants.%Base]
// CHECK:STDOUT:     %self.loc14_8.1: %Base = param self, runtime_param0
// CHECK:STDOUT:     %self.loc14_8.2: %Base = bind_name self, %self.loc14_8.1
// CHECK:STDOUT:     %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc14_25.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:     %.loc14_25.2: type = converted %int.make_type_32.loc14, %.loc14_25.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc14: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.1 = fn_decl @G.1 [template = constants.%G.1] {
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc11 [template = constants.%Derived]
// CHECK:STDOUT:     %self.loc15_8.1: %Derived = param self, runtime_param0
// CHECK:STDOUT:     %self.loc15_8.2: %Derived = bind_name self, %self.loc15_8.1
// CHECK:STDOUT:     %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_28.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:     %.loc15_28.2: type = converted %int.make_type_32.loc15, %.loc15_28.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc15: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[@Base.%self.loc14_8.2: %Base]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.1[@Base.%self.loc15_8.2: %Derived]() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@Derived.%self.loc21_8.2: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2[@Derived.%self.loc22_8.2: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a: %Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @Base.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc26_11: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc26_20.1: ref %Base = class_element_access %a.ref, element0
// CHECK:STDOUT:   %.loc26_20.2: ref %Base = converted %a.ref, %.loc26_20.1
// CHECK:STDOUT:   %.loc26_20.3: %Base = bind_value %.loc26_20.2
// CHECK:STDOUT:   %F.call: init i32 = call %.loc26_11(%.loc26_20.3)
// CHECK:STDOUT:   %.loc26_22.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc26_22.2: i32 = converted %F.call, %.loc26_22.1
// CHECK:STDOUT:   return %.loc26_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect(%p: %.8) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.8 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.1 = name_ref F, @Base.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc30_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc30_11.2: <bound method> = bound_method %.loc30_11.1, %F.ref
// CHECK:STDOUT:   %.loc30_21.1: ref %Base = class_element_access %.loc30_11.1, element0
// CHECK:STDOUT:   %.loc30_21.2: ref %Base = converted %.loc30_11.1, %.loc30_21.1
// CHECK:STDOUT:   %.loc30_21.3: %Base = bind_value %.loc30_21.2
// CHECK:STDOUT:   %F.call: init i32 = call %.loc30_11.2(%.loc30_21.3)
// CHECK:STDOUT:   %.loc30_23.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc30_23.2: i32 = converted %F.call, %.loc30_23.1
// CHECK:STDOUT:   return %.loc30_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBase(%a: %Derived) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.1 = name_ref G, @Base.%G.decl [template = constants.%G.1]
// CHECK:STDOUT:   %.loc34_11: <bound method> = bound_method %a.ref, %G.ref
// CHECK:STDOUT:   %G.call: init i32 = call %.loc34_11(%a.ref)
// CHECK:STDOUT:   %.loc34_22.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc34_22.2: i32 = converted %G.call, %.loc34_22.1
// CHECK:STDOUT:   return %.loc34_22.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBaseIndirect(%p: %.8) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.8 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.1 = name_ref G, @Base.%G.decl [template = constants.%G.1]
// CHECK:STDOUT:   %.loc38_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc38_11.2: <bound method> = bound_method %.loc38_11.1, %G.ref
// CHECK:STDOUT:   %.loc38_11.3: %Derived = bind_value %.loc38_11.1
// CHECK:STDOUT:   %G.call: init i32 = call %.loc38_11.2(%.loc38_11.3)
// CHECK:STDOUT:   %.loc38_23.1: i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc38_23.2: i32 = converted %G.call, %.loc38_23.1
// CHECK:STDOUT:   return %.loc38_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
