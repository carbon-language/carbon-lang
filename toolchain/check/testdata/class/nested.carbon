// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Outer {
  fn F() {
    // Outer and Inner are both complete here.
    var o: Outer;
    var i: Inner;
  }

  class Inner {
    var pi: Self*;
    var po: Outer*;
    var qi: Inner*;

    fn G() {
      // Outer and Inner are both complete here.
      var o: Outer;
      var i: Inner;
    }
  }

  fn H() {
    // Outer and Inner are both complete here.
    var o: Outer;
    var i: Inner;
  }

  var po: Self*;
  var qo: Outer*;
  var pi: Inner*;
}

fn F(a: Outer*) {
  let b: Outer.Inner* = (*a).pi;

  a->po = a;
  a->qo = a;
  a->pi = a->pi;
  b->po = a;
  b->pi = a->pi;
  b->qi = a->pi;
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
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %Outer.ref: type = name_ref Outer, %Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     %.loc37: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:     %a.loc37_6.1: Outer* = param a
// CHECK:STDOUT:     @F.2.%a: Outer* = bind_name a, %a.loc37_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [template = constants.%Inner] {}
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template] {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Outer [template = constants.%Outer]
// CHECK:STDOUT:   %.loc32_15: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc32_9: <unbound element of class Outer> = field_decl po, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %.loc33_16: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc33_9: <unbound element of class Outer> = field_decl qo, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc34_16: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc34_9: <unbound element of class Outer> = field_decl pi, element2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .H = %H
// CHECK:STDOUT:   .po = %.loc32_9
// CHECK:STDOUT:   .qo = %.loc33_9
// CHECK:STDOUT:   .pi = %.loc34_9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [template = constants.%Inner]
// CHECK:STDOUT:   %.loc15_17: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc15_11: <unbound element of class Inner> = field_decl pi, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %.loc16_18: type = ptr_type Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc16_11: <unbound element of class Inner> = field_decl po, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc17_18: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc17_11: <unbound element of class Inner> = field_decl qi, element2 [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc15_11
// CHECK:STDOUT:   .po = %.loc16_11
// CHECK:STDOUT:   .qi = %.loc17_11
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %o.var: ref Outer = var o
// CHECK:STDOUT:   %o: ref Outer = bind_name o, %o.var
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %i.var: ref Inner = var i
// CHECK:STDOUT:   %i: ref Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %o.var: ref Outer = var o
// CHECK:STDOUT:   %o: ref Outer = bind_name o, %o.var
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %i.var: ref Inner = var i
// CHECK:STDOUT:   %i: ref Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %o.var: ref Outer = var o
// CHECK:STDOUT:   %o: ref Outer = bind_name o, %o.var
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %i.var: ref Inner = var i
// CHECK:STDOUT:   %i: ref Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a: Outer*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc38_21: type = ptr_type Inner [template = constants.%.1]
// CHECK:STDOUT:   %a.ref.loc38: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc38_26: ref Outer = deref %a.ref.loc38
// CHECK:STDOUT:   %pi.ref.loc38: <unbound element of class Outer> = name_ref pi, @Outer.%.loc34_9 [template = @Outer.%.loc34_9]
// CHECK:STDOUT:   %.loc38_29.1: ref Inner* = class_element_access %.loc38_26, element2
// CHECK:STDOUT:   %.loc38_29.2: Inner* = bind_value %.loc38_29.1
// CHECK:STDOUT:   %b: Inner* = bind_name b, %.loc38_29.2
// CHECK:STDOUT:   %a.ref.loc40_3: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc40_4.1: ref Outer = deref %a.ref.loc40_3
// CHECK:STDOUT:   %po.ref.loc40: <unbound element of class Outer> = name_ref po, @Outer.%.loc32_9 [template = @Outer.%.loc32_9]
// CHECK:STDOUT:   %.loc40_4.2: ref Outer* = class_element_access %.loc40_4.1, element0
// CHECK:STDOUT:   %a.ref.loc40_11: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc40_4.2, %a.ref.loc40_11
// CHECK:STDOUT:   %a.ref.loc41_3: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc41_4.1: ref Outer = deref %a.ref.loc41_3
// CHECK:STDOUT:   %qo.ref: <unbound element of class Outer> = name_ref qo, @Outer.%.loc33_9 [template = @Outer.%.loc33_9]
// CHECK:STDOUT:   %.loc41_4.2: ref Outer* = class_element_access %.loc41_4.1, element1
// CHECK:STDOUT:   %a.ref.loc41_11: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc41_4.2, %a.ref.loc41_11
// CHECK:STDOUT:   %a.ref.loc42_3: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc42_4.1: ref Outer = deref %a.ref.loc42_3
// CHECK:STDOUT:   %pi.ref.loc42_4: <unbound element of class Outer> = name_ref pi, @Outer.%.loc34_9 [template = @Outer.%.loc34_9]
// CHECK:STDOUT:   %.loc42_4.2: ref Inner* = class_element_access %.loc42_4.1, element2
// CHECK:STDOUT:   %a.ref.loc42_11: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc42_12.1: ref Outer = deref %a.ref.loc42_11
// CHECK:STDOUT:   %pi.ref.loc42_12: <unbound element of class Outer> = name_ref pi, @Outer.%.loc34_9 [template = @Outer.%.loc34_9]
// CHECK:STDOUT:   %.loc42_12.2: ref Inner* = class_element_access %.loc42_12.1, element2
// CHECK:STDOUT:   %.loc42_12.3: Inner* = bind_value %.loc42_12.2
// CHECK:STDOUT:   assign %.loc42_4.2, %.loc42_12.3
// CHECK:STDOUT:   %b.ref.loc43: Inner* = name_ref b, %b
// CHECK:STDOUT:   %.loc43_4.1: ref Inner = deref %b.ref.loc43
// CHECK:STDOUT:   %po.ref.loc43: <unbound element of class Inner> = name_ref po, @Inner.%.loc16_11 [template = @Inner.%.loc16_11]
// CHECK:STDOUT:   %.loc43_4.2: ref Outer* = class_element_access %.loc43_4.1, element1
// CHECK:STDOUT:   %a.ref.loc43: Outer* = name_ref a, %a
// CHECK:STDOUT:   assign %.loc43_4.2, %a.ref.loc43
// CHECK:STDOUT:   %b.ref.loc44: Inner* = name_ref b, %b
// CHECK:STDOUT:   %.loc44_4.1: ref Inner = deref %b.ref.loc44
// CHECK:STDOUT:   %pi.ref.loc44_4: <unbound element of class Inner> = name_ref pi, @Inner.%.loc15_11 [template = @Inner.%.loc15_11]
// CHECK:STDOUT:   %.loc44_4.2: ref Inner* = class_element_access %.loc44_4.1, element0
// CHECK:STDOUT:   %a.ref.loc44: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc44_12.1: ref Outer = deref %a.ref.loc44
// CHECK:STDOUT:   %pi.ref.loc44_12: <unbound element of class Outer> = name_ref pi, @Outer.%.loc34_9 [template = @Outer.%.loc34_9]
// CHECK:STDOUT:   %.loc44_12.2: ref Inner* = class_element_access %.loc44_12.1, element2
// CHECK:STDOUT:   %.loc44_12.3: Inner* = bind_value %.loc44_12.2
// CHECK:STDOUT:   assign %.loc44_4.2, %.loc44_12.3
// CHECK:STDOUT:   %b.ref.loc45: Inner* = name_ref b, %b
// CHECK:STDOUT:   %.loc45_4.1: ref Inner = deref %b.ref.loc45
// CHECK:STDOUT:   %qi.ref: <unbound element of class Inner> = name_ref qi, @Inner.%.loc17_11 [template = @Inner.%.loc17_11]
// CHECK:STDOUT:   %.loc45_4.2: ref Inner* = class_element_access %.loc45_4.1, element2
// CHECK:STDOUT:   %a.ref.loc45: Outer* = name_ref a, %a
// CHECK:STDOUT:   %.loc45_12.1: ref Outer = deref %a.ref.loc45
// CHECK:STDOUT:   %pi.ref.loc45: <unbound element of class Outer> = name_ref pi, @Outer.%.loc34_9 [template = @Outer.%.loc34_9]
// CHECK:STDOUT:   %.loc45_12.2: ref Inner* = class_element_access %.loc45_12.1, element2
// CHECK:STDOUT:   %.loc45_12.3: Inner* = bind_value %.loc45_12.2
// CHECK:STDOUT:   assign %.loc45_4.2, %.loc45_12.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
