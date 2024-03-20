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
  let b: Outer.Inner* = (*a).pi;

  (*a).po = a;
  (*a).qo = a;
  (*a).pi = (*a).pi;
  (*b).po = a;
  (*b).pi = (*a).pi;
  (*b).qi = (*a).pi;
}

// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Outer: type = class_type @Outer [template]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [template]
// CHECK:STDOUT:   %.1: type = ptr_type Inner [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Inner, Inner* [template]
// CHECK:STDOUT:   %.3: type = ptr_type Outer [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type Inner, Outer* [template]
// CHECK:STDOUT:   %.5: type = struct_type {.pi: Inner*, .po: Outer*, .qi: Inner*} [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type Outer, Outer* [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type Outer, Inner* [template]
// CHECK:STDOUT:   %.8: type = struct_type {.po: Outer*, .qo: Outer*, .pi: Inner*} [template]
// CHECK:STDOUT:   %.9: type = ptr_type {.po: Outer*, .qo: Outer*, .pi: Inner*} [template]
// CHECK:STDOUT:   %.10: type = ptr_type {.pi: Inner*, .po: Outer*, .qi: Inner*} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [template = constants.%Outer] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %Outer.ref: type = name_ref Outer, %Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     %.loc19: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:     %a.loc19_6.1: Outer* = param a
// CHECK:STDOUT:     @F.%a: Outer* = bind_name a, %a.loc19_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [template = constants.%Inner] {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Outer [template = constants.%Outer]
// CHECK:STDOUT:   %.loc14_15: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_9: <unbound element of class Outer> = field_decl po, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %.loc15_16: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_9: <unbound element of class Outer> = field_decl qo, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc16_16: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_9: <unbound element of class Outer> = field_decl pi, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .po = %.loc14_9
// CHECK:STDOUT:   .qo = %.loc15_9
// CHECK:STDOUT:   .pi = %.loc16_9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [template = constants.%Inner]
// CHECK:STDOUT:   %.loc9_17: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_11: <unbound element of class Inner> = field_decl pi, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %.loc10_18: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc10_11: <unbound element of class Inner> = field_decl po, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc11_18: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_11: <unbound element of class Inner> = field_decl qi, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc9_11
// CHECK:STDOUT:   .po = %.loc10_11
// CHECK:STDOUT:   .qi = %.loc11_11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: Outer*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc20_21: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %a.ref.loc20: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc20_26: ref Outer = deref %a.ref.loc20
// CHECK:STDOUT:   %pi.ref.loc20: <unbound element of class Outer> = name_ref pi, @Outer.%.loc16_9 [template = @Outer.%.loc16_9]
// CHECK:STDOUT:   %.loc20_29.1: ref Inner* = class_element_access %.loc20_26, element2
// CHECK:STDOUT:   %.loc20_29.2: Inner* = bind_value %.loc20_29.1
// CHECK:STDOUT:   %b: Inner* = bind_name b, %.loc20_29.2
// CHECK:STDOUT:   %a.ref.loc22_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc22_4: ref Outer = deref %a.ref.loc22_5
// CHECK:STDOUT:   %po.ref.loc22: <unbound element of class Outer> = name_ref po, @Outer.%.loc14_9 [template = @Outer.%.loc14_9]
// CHECK:STDOUT:   %.loc22_7: ref Outer* = class_element_access %.loc22_4, element0
// CHECK:STDOUT:   %a.ref.loc22_13: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc22_7, %a.ref.loc22_13
// CHECK:STDOUT:   %a.ref.loc23_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc23_4: ref Outer = deref %a.ref.loc23_5
// CHECK:STDOUT:   %qo.ref: <unbound element of class Outer> = name_ref qo, @Outer.%.loc15_9 [template = @Outer.%.loc15_9]
// CHECK:STDOUT:   %.loc23_7: ref Outer* = class_element_access %.loc23_4, element1
// CHECK:STDOUT:   %a.ref.loc23_13: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc23_7, %a.ref.loc23_13
// CHECK:STDOUT:   %a.ref.loc24_5: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc24_4: ref Outer = deref %a.ref.loc24_5
// CHECK:STDOUT:   %pi.ref.loc24_7: <unbound element of class Outer> = name_ref pi, @Outer.%.loc16_9 [template = @Outer.%.loc16_9]
// CHECK:STDOUT:   %.loc24_7: ref Inner* = class_element_access %.loc24_4, element2
// CHECK:STDOUT:   %a.ref.loc24_15: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc24_14: ref Outer = deref %a.ref.loc24_15
// CHECK:STDOUT:   %pi.ref.loc24_17: <unbound element of class Outer> = name_ref pi, @Outer.%.loc16_9 [template = @Outer.%.loc16_9]
// CHECK:STDOUT:   %.loc24_17.1: ref Inner* = class_element_access %.loc24_14, element2
// CHECK:STDOUT:   %.loc24_17.2: Inner* = bind_value %.loc24_17.1
// CHECK:STDOUT:   assign %.loc24_7, %.loc24_17.2
// CHECK:STDOUT:   %b.ref.loc25: Inner* = name_ref b, %b
// CHECK:STDOUT:   %.loc25_4: ref Inner = deref %b.ref.loc25
// CHECK:STDOUT:   %po.ref.loc25: <unbound element of class Inner> = name_ref po, @Inner.%.loc10_11 [template = @Inner.%.loc10_11]
// CHECK:STDOUT:   %.loc25_7: ref Outer* = class_element_access %.loc25_4, element1
// CHECK:STDOUT:   %a.ref.loc25: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc25_7, %a.ref.loc25
// CHECK:STDOUT:   %b.ref.loc26: Inner* = name_ref b, %b
// CHECK:STDOUT:   %.loc26_4: ref Inner = deref %b.ref.loc26
// CHECK:STDOUT:   %pi.ref.loc26_7: <unbound element of class Inner> = name_ref pi, @Inner.%.loc9_11 [template = @Inner.%.loc9_11]
// CHECK:STDOUT:   %.loc26_7: ref Inner* = class_element_access %.loc26_4, element0
// CHECK:STDOUT:   %a.ref.loc26: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc26_14: ref Outer = deref %a.ref.loc26
// CHECK:STDOUT:   %pi.ref.loc26_17: <unbound element of class Outer> = name_ref pi, @Outer.%.loc16_9 [template = @Outer.%.loc16_9]
// CHECK:STDOUT:   %.loc26_17.1: ref Inner* = class_element_access %.loc26_14, element2
// CHECK:STDOUT:   %.loc26_17.2: Inner* = bind_value %.loc26_17.1
// CHECK:STDOUT:   assign %.loc26_7, %.loc26_17.2
// CHECK:STDOUT:   %b.ref.loc27: Inner* = name_ref b, %b
// CHECK:STDOUT:   %.loc27_4: ref Inner = deref %b.ref.loc27
// CHECK:STDOUT:   %qi.ref: <unbound element of class Inner> = name_ref qi, @Inner.%.loc11_11 [template = @Inner.%.loc11_11]
// CHECK:STDOUT:   %.loc27_7: ref Inner* = class_element_access %.loc27_4, element2
// CHECK:STDOUT:   %a.ref.loc27: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc27_14: ref Outer = deref %a.ref.loc27
// CHECK:STDOUT:   %pi.ref.loc27: <unbound element of class Outer> = name_ref pi, @Outer.%.loc16_9 [template = @Outer.%.loc16_9]
// CHECK:STDOUT:   %.loc27_17.1: ref Inner* = class_element_access %.loc27_14, element2
// CHECK:STDOUT:   %.loc27_17.2: Inner* = bind_value %.loc27_17.1
// CHECK:STDOUT:   assign %.loc27_7, %.loc27_17.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
