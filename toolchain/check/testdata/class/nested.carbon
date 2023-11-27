// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Outer {
  class Inner {
    var pi: Self*;
    var po: Outer*;
    var qi: Inner*;
  }

  var po: Self*;
  var qo: Outer*;
  var pi: Inner*;
}

fn F(a: Outer*) {
  // TODO: Simplify this once `Outer.Inner` works.
  // let b: Outer.Inner* = (*a).pi;

  (*a).po = a;
  (*a).qo = a;
  (*a).pi = (*a).pi;
  (*(*a).pi).po = a;
  (*(*a).pi).pi = (*a).pi;
  (*(*a).pi).qi = (*a).pi;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc12_3.1: type = struct_type {.pi: Inner*, .po: Outer*, .qi: Inner*}
// CHECK:STDOUT:   %.loc17_1.1: type = struct_type {.po: Outer*, .qo: Outer*, .pi: Inner*}
// CHECK:STDOUT:   %.loc17_1.2: type = ptr_type {.po: Outer*, .qo: Outer*, .pi: Inner*}
// CHECK:STDOUT:   %.loc12_3.2: type = ptr_type {.pi: Inner*, .po: Outer*, .qi: Inner*}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "nested.carbon" {
// CHECK:STDOUT:   class_decl @Outer, ()
// CHECK:STDOUT:   %Outer: type = class_type @Outer
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   class_decl @Inner, ()
// CHECK:STDOUT:   %Inner: type = class_type @Inner
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, file.%Outer
// CHECK:STDOUT:   %.loc14_15: type = ptr_type Outer
// CHECK:STDOUT:   %.loc14_9.1: type = unbound_field_type Outer, Outer*
// CHECK:STDOUT:   %.loc14_9.2: <unbound field of class Outer> = field po, member0
// CHECK:STDOUT:   %po: <unbound field of class Outer> = bind_name po, %.loc14_9.2
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer
// CHECK:STDOUT:   %.loc15_16: type = ptr_type Outer
// CHECK:STDOUT:   %.loc15_9.1: type = unbound_field_type Outer, Outer*
// CHECK:STDOUT:   %.loc15_9.2: <unbound field of class Outer> = field qo, member1
// CHECK:STDOUT:   %qo: <unbound field of class Outer> = bind_name qo, %.loc15_9.2
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner
// CHECK:STDOUT:   %.loc16_16: type = ptr_type Inner
// CHECK:STDOUT:   %.loc16_9.1: type = unbound_field_type Outer, Inner*
// CHECK:STDOUT:   %.loc16_9.2: <unbound field of class Outer> = field pi, member2
// CHECK:STDOUT:   %pi: <unbound field of class Outer> = bind_name pi, %.loc16_9.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Inner = <unexpected instref inst+2>
// CHECK:STDOUT:   .po = %po
// CHECK:STDOUT:   .qo = %qo
// CHECK:STDOUT:   .pi = %pi
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @Outer.%Inner
// CHECK:STDOUT:   %.loc9_17: type = ptr_type Inner
// CHECK:STDOUT:   %.loc9_11.1: type = unbound_field_type Inner, Inner*
// CHECK:STDOUT:   %.loc9_11.2: <unbound field of class Inner> = field pi, member0
// CHECK:STDOUT:   %pi: <unbound field of class Inner> = bind_name pi, %.loc9_11.2
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer
// CHECK:STDOUT:   %.loc10_18: type = ptr_type Outer
// CHECK:STDOUT:   %.loc10_11.1: type = unbound_field_type Inner, Outer*
// CHECK:STDOUT:   %.loc10_11.2: <unbound field of class Inner> = field po, member1
// CHECK:STDOUT:   %po: <unbound field of class Inner> = bind_name po, %.loc10_11.2
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner
// CHECK:STDOUT:   %.loc11_18: type = ptr_type Inner
// CHECK:STDOUT:   %.loc11_11.1: type = unbound_field_type Inner, Inner*
// CHECK:STDOUT:   %.loc11_11.2: <unbound field of class Inner> = field qi, member2
// CHECK:STDOUT:   %qi: <unbound field of class Inner> = bind_name qi, %.loc11_11.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .pi = %pi
// CHECK:STDOUT:   .po = %po
// CHECK:STDOUT:   .qi = %qi
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: Outer*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref.loc23_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc23_4: ref Outer = deref %a.ref.loc23_5
// CHECK:STDOUT:   %.loc23_7: ref Outer* = class_field_access %.loc23_4, member0
// CHECK:STDOUT:   %a.ref.loc23_13: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc23_7, %a.ref.loc23_13
// CHECK:STDOUT:   %a.ref.loc24_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc24_4: ref Outer = deref %a.ref.loc24_5
// CHECK:STDOUT:   %.loc24_7: ref Outer* = class_field_access %.loc24_4, member1
// CHECK:STDOUT:   %a.ref.loc24_13: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc24_7, %a.ref.loc24_13
// CHECK:STDOUT:   %a.ref.loc25_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc25_4: ref Outer = deref %a.ref.loc25_5
// CHECK:STDOUT:   %.loc25_7: ref Inner* = class_field_access %.loc25_4, member2
// CHECK:STDOUT:   %a.ref.loc25_15: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc25_14: ref Outer = deref %a.ref.loc25_15
// CHECK:STDOUT:   %.loc25_17.1: ref Inner* = class_field_access %.loc25_14, member2
// CHECK:STDOUT:   %.loc25_17.2: Inner* = bind_value %.loc25_17.1
// CHECK:STDOUT:   assign %.loc25_7, %.loc25_17.2
// CHECK:STDOUT:   %a.ref.loc26_7: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc26_6: ref Outer = deref %a.ref.loc26_7
// CHECK:STDOUT:   %.loc26_9.1: ref Inner* = class_field_access %.loc26_6, member2
// CHECK:STDOUT:   %.loc26_9.2: Inner* = bind_value %.loc26_9.1
// CHECK:STDOUT:   %.loc26_4: ref Inner = deref %.loc26_9.2
// CHECK:STDOUT:   %.loc26_13: ref Outer* = class_field_access %.loc26_4, member1
// CHECK:STDOUT:   %a.ref.loc26_19: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc26_13, %a.ref.loc26_19
// CHECK:STDOUT:   %a.ref.loc27_7: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc27_6: ref Outer = deref %a.ref.loc27_7
// CHECK:STDOUT:   %.loc27_9.1: ref Inner* = class_field_access %.loc27_6, member2
// CHECK:STDOUT:   %.loc27_9.2: Inner* = bind_value %.loc27_9.1
// CHECK:STDOUT:   %.loc27_4: ref Inner = deref %.loc27_9.2
// CHECK:STDOUT:   %.loc27_13: ref Inner* = class_field_access %.loc27_4, member0
// CHECK:STDOUT:   %a.ref.loc27_21: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc27_20: ref Outer = deref %a.ref.loc27_21
// CHECK:STDOUT:   %.loc27_23.1: ref Inner* = class_field_access %.loc27_20, member2
// CHECK:STDOUT:   %.loc27_23.2: Inner* = bind_value %.loc27_23.1
// CHECK:STDOUT:   assign %.loc27_13, %.loc27_23.2
// CHECK:STDOUT:   %a.ref.loc28_7: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc28_6: ref Outer = deref %a.ref.loc28_7
// CHECK:STDOUT:   %.loc28_9.1: ref Inner* = class_field_access %.loc28_6, member2
// CHECK:STDOUT:   %.loc28_9.2: Inner* = bind_value %.loc28_9.1
// CHECK:STDOUT:   %.loc28_4: ref Inner = deref %.loc28_9.2
// CHECK:STDOUT:   %.loc28_13: ref Inner* = class_field_access %.loc28_4, member2
// CHECK:STDOUT:   %a.ref.loc28_21: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc28_20: ref Outer = deref %a.ref.loc28_21
// CHECK:STDOUT:   %.loc28_23.1: ref Inner* = class_field_access %.loc28_20, member2
// CHECK:STDOUT:   %.loc28_23.2: Inner* = bind_value %.loc28_23.1
// CHECK:STDOUT:   assign %.loc28_13, %.loc28_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
