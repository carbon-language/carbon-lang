// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/base_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/base_function.carbon

// --- base_method.carbon

library "[[@TEST_NAME]]";

base class Base {
  var a: {};

  fn F[addr self: Self*]();
}

fn Base.F[addr self: Self*]() {
  (*self).a = {};
}

class Derived {
  extend base: Base;
}

fn Call(p: Derived*) {
  //@dump-sem-ir-begin
  (*p).F();
  //@dump-sem-ir-end
}

// --- base_method_shadow.carbon

library "[[@TEST_NAME]]";

base class A {
  fn F[addr self: Self*]();
}

base class B {
  extend base: A;
  fn F[addr self: Self*]();
}

class C {
  extend base: B;
  fn F[addr self: Self*]();
}

class D {
  extend base: B;
}

fn Call(a: A*, b: B*, c: C*, d: D*) {
  //@dump-sem-ir-begin
  (*a).F();
  (*b).F();
  (*c).F();
  (*d).F();
  //@dump-sem-ir-end
}

// --- base_method_qualified.carbon

library "[[@TEST_NAME]]";

class Derived;

base class Base {
  fn F[self: Self]() -> {};
  fn G[self: Derived]() -> {};
}

class Derived {
  extend base: Base;

  fn F[self: Self]();
  fn G[self: Self]();
}

//@dump-sem-ir-begin
fn Call(a: Derived) -> {} {
  return a.(Base.F)();
}

fn CallIndirect(p: Derived*) -> {} {
  return p->(Base.F)();
}

fn PassDerivedToBase(a: Derived) -> {} {
  return a.(Base.G)();
}

fn PassDerivedToBaseIndirect(p: Derived*) -> {} {
  return p->(Base.G)();
}
//@dump-sem-ir-end

// --- base_function_unqualified.carbon

library "[[@TEST_NAME]]";

base class Base {
  fn F();
}

class Derived {
  extend base: Base;

  fn G() { F(); }
  fn H();
}

fn Derived.H() {
  //@dump-sem-ir-begin
  F();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- base_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%p.param: %ptr.404) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %.loc20_4.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Base.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc20_4.1, %F.ref
// CHECK:STDOUT:   %addr.loc20_4.1: %ptr.404 = addr_of %.loc20_4.1
// CHECK:STDOUT:   %.loc20_4.2: ref %Derived = deref %addr.loc20_4.1
// CHECK:STDOUT:   %.loc20_4.3: ref %Base = class_element_access %.loc20_4.2, element0
// CHECK:STDOUT:   %addr.loc20_4.2: %ptr.11f = addr_of %.loc20_4.3
// CHECK:STDOUT:   %.loc20_4.4: %ptr.11f = converted %addr.loc20_4.1, %addr.loc20_4.2
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.bound(%.loc20_4.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_method_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %F.type.649: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.485: %F.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %F.type.8c6: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.92a: %F.type.8c6 = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %F.type.c29: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.437: %F.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param: %ptr.6db, %b.param: %ptr.e79, %c.param: %ptr.019, %d.param: %ptr.19c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.6db = name_ref a, %a
// CHECK:STDOUT:   %.loc24: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc24: %F.type.649 = name_ref F, @A.%F.decl [concrete = constants.%F.485]
// CHECK:STDOUT:   %F.bound.loc24: <bound method> = bound_method %.loc24, %F.ref.loc24
// CHECK:STDOUT:   %addr.loc24: %ptr.6db = addr_of %.loc24
// CHECK:STDOUT:   %F.call.loc24: init %empty_tuple.type = call %F.bound.loc24(%addr.loc24)
// CHECK:STDOUT:   %b.ref: %ptr.e79 = name_ref b, %b
// CHECK:STDOUT:   %.loc25: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc25: %F.type.8c6 = name_ref F, @B.%F.decl [concrete = constants.%F.92a]
// CHECK:STDOUT:   %F.bound.loc25: <bound method> = bound_method %.loc25, %F.ref.loc25
// CHECK:STDOUT:   %addr.loc25: %ptr.e79 = addr_of %.loc25
// CHECK:STDOUT:   %F.call.loc25: init %empty_tuple.type = call %F.bound.loc25(%addr.loc25)
// CHECK:STDOUT:   %c.ref: %ptr.019 = name_ref c, %c
// CHECK:STDOUT:   %.loc26: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc26: %F.type.c29 = name_ref F, @C.%F.decl [concrete = constants.%F.437]
// CHECK:STDOUT:   %F.bound.loc26: <bound method> = bound_method %.loc26, %F.ref.loc26
// CHECK:STDOUT:   %addr.loc26: %ptr.019 = addr_of %.loc26
// CHECK:STDOUT:   %F.call.loc26: init %empty_tuple.type = call %F.bound.loc26(%addr.loc26)
// CHECK:STDOUT:   %d.ref: %ptr.19c = name_ref d, %d
// CHECK:STDOUT:   %.loc27_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc27: %F.type.8c6 = name_ref F, @B.%F.decl [concrete = constants.%F.92a]
// CHECK:STDOUT:   %F.bound.loc27: <bound method> = bound_method %.loc27_4.1, %F.ref.loc27
// CHECK:STDOUT:   %addr.loc27_4.1: %ptr.19c = addr_of %.loc27_4.1
// CHECK:STDOUT:   %.loc27_4.2: ref %D = deref %addr.loc27_4.1
// CHECK:STDOUT:   %.loc27_4.3: ref %B = class_element_access %.loc27_4.2, element0
// CHECK:STDOUT:   %addr.loc27_4.2: %ptr.e79 = addr_of %.loc27_4.3
// CHECK:STDOUT:   %.loc27_4.4: %ptr.e79 = converted %addr.loc27_4.1, %addr.loc27_4.2
// CHECK:STDOUT:   %F.call.loc27: init %empty_tuple.type = call %F.bound.loc27(%.loc27_4.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_method_qualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %G.type.6ee: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.663: %G.type.6ee = struct_value () [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.605: type = pattern_type %ptr.404 [concrete]
// CHECK:STDOUT:   %CallIndirect.type: type = fn_type @CallIndirect [concrete]
// CHECK:STDOUT:   %CallIndirect: %CallIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassDerivedToBase.type: type = fn_type @PassDerivedToBase [concrete]
// CHECK:STDOUT:   %PassDerivedToBase: %PassDerivedToBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.type: type = fn_type @PassDerivedToBaseIndirect [concrete]
// CHECK:STDOUT:   %PassDerivedToBaseIndirect: %PassDerivedToBaseIndirect.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %a.patt: %pattern_type.fb9 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.fb9 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_25.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc19_25.2: type = converted %.loc19_25.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %a.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc4 [concrete = constants.%Derived]
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %CallIndirect.decl: %CallIndirect.type = fn_decl @CallIndirect [concrete = constants.%CallIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc23_34.2: type = converted %.loc23_34.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc23_27: type = splice_block %ptr [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc4 [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBase.decl: %PassDerivedToBase.type = fn_decl @PassDerivedToBase [concrete = constants.%PassDerivedToBase] {
// CHECK:STDOUT:     %a.patt: %pattern_type.fb9 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.fb9 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc27_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc27_38.2: type = converted %.loc27_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %a.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc4 [concrete = constants.%Derived]
// CHECK:STDOUT:     %a: %Derived = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassDerivedToBaseIndirect.decl: %PassDerivedToBaseIndirect.type = fn_decl @PassDerivedToBaseIndirect [concrete = constants.%PassDerivedToBaseIndirect] {
// CHECK:STDOUT:     %p.patt: %pattern_type.605 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.605 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc31_47.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc31_47.2: type = converted %.loc31_47.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %p.param: %ptr.404 = value_param call_param0
// CHECK:STDOUT:     %.loc31_40: type = splice_block %ptr [concrete = constants.%ptr.404] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl.loc4 [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr: type = ptr_type %Derived.ref [concrete = constants.%ptr.404]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.404 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param1
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param: %Derived) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.7c6 = name_ref F, @Base.%F.decl [concrete = constants.%F.d17]
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %a.ref, %F.ref
// CHECK:STDOUT:   %.loc20_10.1: ref %Base = class_element_access %a.ref, element0
// CHECK:STDOUT:   %.loc20_10.2: ref %Base = converted %a.ref, %.loc20_10.1
// CHECK:STDOUT:   %.loc20_10.3: %Base = bind_value %.loc20_10.2
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.bound(%.loc20_10.3)
// CHECK:STDOUT:   %.loc20_21.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc20_21.2: ref %empty_struct_type = temporary %.loc20_21.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_22: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc20_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallIndirect(%p.param: %ptr.404) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %F.ref: %F.type.7c6 = name_ref F, @Base.%F.decl [concrete = constants.%F.d17]
// CHECK:STDOUT:   %.loc24_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %F.bound: <bound method> = bound_method %.loc24_11.1, %F.ref
// CHECK:STDOUT:   %.loc24_11.2: ref %Base = class_element_access %.loc24_11.1, element0
// CHECK:STDOUT:   %.loc24_11.3: ref %Base = converted %.loc24_11.1, %.loc24_11.2
// CHECK:STDOUT:   %.loc24_11.4: %Base = bind_value %.loc24_11.3
// CHECK:STDOUT:   %F.call: init %empty_struct_type = call %F.bound(%.loc24_11.4)
// CHECK:STDOUT:   %.loc24_22.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc24_22.2: ref %empty_struct_type = temporary %.loc24_22.1, %F.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc24_23: %empty_struct_type = converted %F.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc24_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBase(%a.param: %Derived) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Derived = name_ref a, %a
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.6ee = name_ref G, @Base.%G.decl [concrete = constants.%G.663]
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %a.ref, %G.ref
// CHECK:STDOUT:   %G.call: init %empty_struct_type = call %G.bound(%a.ref)
// CHECK:STDOUT:   %.loc28_21.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc28_21.2: ref %empty_struct_type = temporary %.loc28_21.1, %G.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc28_22: %empty_struct_type = converted %G.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc28_22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassDerivedToBaseIndirect(%p.param: %ptr.404) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.404 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %G.ref: %G.type.6ee = name_ref G, @Base.%G.decl [concrete = constants.%G.663]
// CHECK:STDOUT:   %.loc32_11.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %G.bound: <bound method> = bound_method %.loc32_11.1, %G.ref
// CHECK:STDOUT:   %.loc32_11.2: %Derived = bind_value %.loc32_11.1
// CHECK:STDOUT:   %G.call: init %empty_struct_type = call %G.bound(%.loc32_11.2)
// CHECK:STDOUT:   %.loc32_22.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc32_22.2: ref %empty_struct_type = temporary %.loc32_22.1, %G.call
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc32_23: %empty_struct_type = converted %G.call, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc32_23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- base_function_unqualified.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, @Base.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
