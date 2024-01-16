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

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {.pi: Inner*, .po: Outer*, .qi: Inner*} [template]
// CHECK:STDOUT:   %.2: type = struct_type {.po: Outer*, .qo: Outer*, .pi: Inner*} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {.po: Outer*, .qo: Outer*, .pi: Inner*} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.pi: Inner*, .po: Outer*, .qi: Inner*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Outer = %Outer.decl, .F = %F}
// CHECK:STDOUT:   %Outer.decl = class_decl @Outer, ()
// CHECK:STDOUT:   %Outer: type = class_type @Outer [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.decl = class_decl @Inner, ()
// CHECK:STDOUT:   %Inner: type = class_type @Inner [template]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, file.%Outer [template = file.%Outer]
// CHECK:STDOUT:   %.loc14_15: type = ptr_type Outer [template]
// CHECK:STDOUT:   %.loc14_9.1: type = unbound_element_type Outer, Outer* [template]
// CHECK:STDOUT:   %.loc14_9.2: <unbound element of class Outer> = field_decl po, element0 [template]
// CHECK:STDOUT:   %po: <unbound element of class Outer> = bind_name po, %.loc14_9.2 [template = %.loc14_9.2]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer [template = file.%Outer]
// CHECK:STDOUT:   %.loc15_16: type = ptr_type Outer [template]
// CHECK:STDOUT:   %.loc15_9.1: type = unbound_element_type Outer, Outer* [template]
// CHECK:STDOUT:   %.loc15_9.2: <unbound element of class Outer> = field_decl qo, element1 [template]
// CHECK:STDOUT:   %qo: <unbound element of class Outer> = bind_name qo, %.loc15_9.2 [template = %.loc15_9.2]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner [template = %Inner]
// CHECK:STDOUT:   %.loc16_16: type = ptr_type Inner [template]
// CHECK:STDOUT:   %.loc16_9.1: type = unbound_element_type Outer, Inner* [template]
// CHECK:STDOUT:   %.loc16_9.2: <unbound element of class Outer> = field_decl pi, element2 [template]
// CHECK:STDOUT:   %pi: <unbound element of class Outer> = bind_name pi, %.loc16_9.2 [template = %.loc16_9.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .po = %po
// CHECK:STDOUT:   .qo = %qo
// CHECK:STDOUT:   .pi = %pi
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @Outer.%Inner [template = @Outer.%Inner]
// CHECK:STDOUT:   %.loc9_17: type = ptr_type Inner [template]
// CHECK:STDOUT:   %.loc9_11.1: type = unbound_element_type Inner, Inner* [template]
// CHECK:STDOUT:   %.loc9_11.2: <unbound element of class Inner> = field_decl pi, element0 [template]
// CHECK:STDOUT:   %pi: <unbound element of class Inner> = bind_name pi, %.loc9_11.2 [template = %.loc9_11.2]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer [template = file.%Outer]
// CHECK:STDOUT:   %.loc10_18: type = ptr_type Outer [template]
// CHECK:STDOUT:   %.loc10_11.1: type = unbound_element_type Inner, Outer* [template]
// CHECK:STDOUT:   %.loc10_11.2: <unbound element of class Inner> = field_decl po, element1 [template]
// CHECK:STDOUT:   %po: <unbound element of class Inner> = bind_name po, %.loc10_11.2 [template = %.loc10_11.2]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner [template = @Outer.%Inner]
// CHECK:STDOUT:   %.loc11_18: type = ptr_type Inner [template]
// CHECK:STDOUT:   %.loc11_11.1: type = unbound_element_type Inner, Inner* [template]
// CHECK:STDOUT:   %.loc11_11.2: <unbound element of class Inner> = field_decl qi, element2 [template]
// CHECK:STDOUT:   %qi: <unbound element of class Inner> = bind_name qi, %.loc11_11.2 [template = %.loc11_11.2]
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
// CHECK:STDOUT:   %.loc23_7: ref Outer* = class_element_access %.loc23_4, element0
// CHECK:STDOUT:   %a.ref.loc23_13: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc23_7, %a.ref.loc23_13
// CHECK:STDOUT:   %a.ref.loc24_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc24_4: ref Outer = deref %a.ref.loc24_5
// CHECK:STDOUT:   %.loc24_7: ref Outer* = class_element_access %.loc24_4, element1
// CHECK:STDOUT:   %a.ref.loc24_13: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc24_7, %a.ref.loc24_13
// CHECK:STDOUT:   %a.ref.loc25_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc25_4: ref Outer = deref %a.ref.loc25_5
// CHECK:STDOUT:   %.loc25_7: ref Inner* = class_element_access %.loc25_4, element2
// CHECK:STDOUT:   %a.ref.loc25_15: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc25_14: ref Outer = deref %a.ref.loc25_15
// CHECK:STDOUT:   %.loc25_17.1: ref Inner* = class_element_access %.loc25_14, element2
// CHECK:STDOUT:   %.loc25_17.2: Inner* = bind_value %.loc25_17.1
// CHECK:STDOUT:   assign %.loc25_7, %.loc25_17.2
// CHECK:STDOUT:   %a.ref.loc26_7: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc26_6: ref Outer = deref %a.ref.loc26_7
// CHECK:STDOUT:   %.loc26_9.1: ref Inner* = class_element_access %.loc26_6, element2
// CHECK:STDOUT:   %.loc26_9.2: Inner* = bind_value %.loc26_9.1
// CHECK:STDOUT:   %.loc26_4: ref Inner = deref %.loc26_9.2
// CHECK:STDOUT:   %.loc26_13: ref Outer* = class_element_access %.loc26_4, element1
// CHECK:STDOUT:   %a.ref.loc26_19: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc26_13, %a.ref.loc26_19
// CHECK:STDOUT:   %a.ref.loc27_7: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc27_6: ref Outer = deref %a.ref.loc27_7
// CHECK:STDOUT:   %.loc27_9.1: ref Inner* = class_element_access %.loc27_6, element2
// CHECK:STDOUT:   %.loc27_9.2: Inner* = bind_value %.loc27_9.1
// CHECK:STDOUT:   %.loc27_4: ref Inner = deref %.loc27_9.2
// CHECK:STDOUT:   %.loc27_13: ref Inner* = class_element_access %.loc27_4, element0
// CHECK:STDOUT:   %a.ref.loc27_21: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc27_20: ref Outer = deref %a.ref.loc27_21
// CHECK:STDOUT:   %.loc27_23.1: ref Inner* = class_element_access %.loc27_20, element2
// CHECK:STDOUT:   %.loc27_23.2: Inner* = bind_value %.loc27_23.1
// CHECK:STDOUT:   assign %.loc27_13, %.loc27_23.2
// CHECK:STDOUT:   %a.ref.loc28_7: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc28_6: ref Outer = deref %a.ref.loc28_7
// CHECK:STDOUT:   %.loc28_9.1: ref Inner* = class_element_access %.loc28_6, element2
// CHECK:STDOUT:   %.loc28_9.2: Inner* = bind_value %.loc28_9.1
// CHECK:STDOUT:   %.loc28_4: ref Inner = deref %.loc28_9.2
// CHECK:STDOUT:   %.loc28_13: ref Inner* = class_element_access %.loc28_4, element2
// CHECK:STDOUT:   %a.ref.loc28_21: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc28_20: ref Outer = deref %a.ref.loc28_21
// CHECK:STDOUT:   %.loc28_23.1: ref Inner* = class_element_access %.loc28_20, element2
// CHECK:STDOUT:   %.loc28_23.2: Inner* = bind_value %.loc28_23.1
// CHECK:STDOUT:   assign %.loc28_13, %.loc28_23.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
