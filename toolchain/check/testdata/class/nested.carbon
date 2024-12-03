// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/nested.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/nested.carbon

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
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %Inner [template]
// CHECK:STDOUT:   %Inner.elem.1: type = unbound_element_type %Inner, %ptr.1 [template]
// CHECK:STDOUT:   %ptr.2: type = ptr_type %Outer [template]
// CHECK:STDOUT:   %Inner.elem.2: type = unbound_element_type %Inner, %ptr.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.pi.po.qi: type = struct_type {.pi: %ptr.1, .po: %ptr.2, .qi: %ptr.1} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %struct_type.pi.po.qi [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.elem.1: type = unbound_element_type %Outer, %ptr.2 [template]
// CHECK:STDOUT:   %Outer.elem.2: type = unbound_element_type %Outer, %ptr.1 [template]
// CHECK:STDOUT:   %struct_type.po.qo.pi: type = struct_type {.po: %ptr.2, .qo: %ptr.2, .pi: %ptr.1} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.po.qo.pi [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [template = constants.%Outer] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %a.patt: %ptr.2 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.2 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Outer.ref.loc41: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     %ptr.loc41: type = ptr_type %Outer [template = constants.%ptr.2]
// CHECK:STDOUT:     %a.param: %ptr.2 = value_param runtime_param0
// CHECK:STDOUT:     %a: %ptr.2 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [template = constants.%Inner] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Outer [template = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc36: type = ptr_type %Outer [template = constants.%ptr.2]
// CHECK:STDOUT:   %.loc36: %Outer.elem.1 = field_decl po, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc37: type = ptr_type %Outer [template = constants.%ptr.2]
// CHECK:STDOUT:   %.loc37: %Outer.elem.1 = field_decl qo, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc38: type = ptr_type %Inner [template = constants.%ptr.1]
// CHECK:STDOUT:   %.loc38: %Outer.elem.2 = field_decl pi, element2 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.po.qo.pi [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .po = %.loc36
// CHECK:STDOUT:   .qo = %.loc37
// CHECK:STDOUT:   .pi = %.loc38
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [template = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc19: type = ptr_type %Inner [template = constants.%ptr.1]
// CHECK:STDOUT:   %.loc19: %Inner.elem.1 = field_decl pi, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc20: type = ptr_type %Outer [template = constants.%ptr.2]
// CHECK:STDOUT:   %.loc20: %Inner.elem.2 = field_decl po, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc21: type = ptr_type %Inner [template = constants.%ptr.1]
// CHECK:STDOUT:   %.loc21: %Inner.elem.1 = field_decl qi, element2 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.pi.po.qi [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc19
// CHECK:STDOUT:   .po = %.loc20
// CHECK:STDOUT:   .qi = %.loc21
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %o.var: ref %Outer = var o
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %i.var: ref %Inner = var i
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %o.var: ref %Outer = var o
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %i.var: ref %Inner = var i
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %o.var: ref %Outer = var o
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %i.var: ref %Inner = var i
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param_patt: %ptr.2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref.loc42: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc42: type = ptr_type %Inner [template = constants.%ptr.1]
// CHECK:STDOUT:   %a.ref.loc42: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc42_26: ref %Outer = deref %a.ref.loc42
// CHECK:STDOUT:   %pi.ref.loc42: %Outer.elem.2 = name_ref pi, @Outer.%.loc38 [template = @Outer.%.loc38]
// CHECK:STDOUT:   %.loc42_29.1: ref %ptr.1 = class_element_access %.loc42_26, element2
// CHECK:STDOUT:   %.loc42_29.2: %ptr.1 = bind_value %.loc42_29.1
// CHECK:STDOUT:   %b: %ptr.1 = bind_name b, %.loc42_29.2
// CHECK:STDOUT:   %a.ref.loc44_3: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc44_4.1: ref %Outer = deref %a.ref.loc44_3
// CHECK:STDOUT:   %po.ref.loc44: %Outer.elem.1 = name_ref po, @Outer.%.loc36 [template = @Outer.%.loc36]
// CHECK:STDOUT:   %.loc44_4.2: ref %ptr.2 = class_element_access %.loc44_4.1, element0
// CHECK:STDOUT:   %a.ref.loc44_11: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   assign %.loc44_4.2, %a.ref.loc44_11
// CHECK:STDOUT:   %a.ref.loc45_3: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc45_4.1: ref %Outer = deref %a.ref.loc45_3
// CHECK:STDOUT:   %qo.ref: %Outer.elem.1 = name_ref qo, @Outer.%.loc37 [template = @Outer.%.loc37]
// CHECK:STDOUT:   %.loc45_4.2: ref %ptr.2 = class_element_access %.loc45_4.1, element1
// CHECK:STDOUT:   %a.ref.loc45_11: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   assign %.loc45_4.2, %a.ref.loc45_11
// CHECK:STDOUT:   %a.ref.loc46_3: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc46_4.1: ref %Outer = deref %a.ref.loc46_3
// CHECK:STDOUT:   %pi.ref.loc46_4: %Outer.elem.2 = name_ref pi, @Outer.%.loc38 [template = @Outer.%.loc38]
// CHECK:STDOUT:   %.loc46_4.2: ref %ptr.1 = class_element_access %.loc46_4.1, element2
// CHECK:STDOUT:   %a.ref.loc46_11: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc46_12.1: ref %Outer = deref %a.ref.loc46_11
// CHECK:STDOUT:   %pi.ref.loc46_12: %Outer.elem.2 = name_ref pi, @Outer.%.loc38 [template = @Outer.%.loc38]
// CHECK:STDOUT:   %.loc46_12.2: ref %ptr.1 = class_element_access %.loc46_12.1, element2
// CHECK:STDOUT:   %.loc46_12.3: %ptr.1 = bind_value %.loc46_12.2
// CHECK:STDOUT:   assign %.loc46_4.2, %.loc46_12.3
// CHECK:STDOUT:   %b.ref.loc47: %ptr.1 = name_ref b, %b
// CHECK:STDOUT:   %.loc47_4.1: ref %Inner = deref %b.ref.loc47
// CHECK:STDOUT:   %po.ref.loc47: %Inner.elem.2 = name_ref po, @Inner.%.loc20 [template = @Inner.%.loc20]
// CHECK:STDOUT:   %.loc47_4.2: ref %ptr.2 = class_element_access %.loc47_4.1, element1
// CHECK:STDOUT:   %a.ref.loc47: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   assign %.loc47_4.2, %a.ref.loc47
// CHECK:STDOUT:   %b.ref.loc48: %ptr.1 = name_ref b, %b
// CHECK:STDOUT:   %.loc48_4.1: ref %Inner = deref %b.ref.loc48
// CHECK:STDOUT:   %pi.ref.loc48_4: %Inner.elem.1 = name_ref pi, @Inner.%.loc19 [template = @Inner.%.loc19]
// CHECK:STDOUT:   %.loc48_4.2: ref %ptr.1 = class_element_access %.loc48_4.1, element0
// CHECK:STDOUT:   %a.ref.loc48: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc48_12.1: ref %Outer = deref %a.ref.loc48
// CHECK:STDOUT:   %pi.ref.loc48_12: %Outer.elem.2 = name_ref pi, @Outer.%.loc38 [template = @Outer.%.loc38]
// CHECK:STDOUT:   %.loc48_12.2: ref %ptr.1 = class_element_access %.loc48_12.1, element2
// CHECK:STDOUT:   %.loc48_12.3: %ptr.1 = bind_value %.loc48_12.2
// CHECK:STDOUT:   assign %.loc48_4.2, %.loc48_12.3
// CHECK:STDOUT:   %b.ref.loc49: %ptr.1 = name_ref b, %b
// CHECK:STDOUT:   %.loc49_4.1: ref %Inner = deref %b.ref.loc49
// CHECK:STDOUT:   %qi.ref: %Inner.elem.1 = name_ref qi, @Inner.%.loc21 [template = @Inner.%.loc21]
// CHECK:STDOUT:   %.loc49_4.2: ref %ptr.1 = class_element_access %.loc49_4.1, element2
// CHECK:STDOUT:   %a.ref.loc49: %ptr.2 = name_ref a, %a
// CHECK:STDOUT:   %.loc49_12.1: ref %Outer = deref %a.ref.loc49
// CHECK:STDOUT:   %pi.ref.loc49: %Outer.elem.2 = name_ref pi, @Outer.%.loc38 [template = @Outer.%.loc38]
// CHECK:STDOUT:   %.loc49_12.2: ref %ptr.1 = class_element_access %.loc49_12.1, element2
// CHECK:STDOUT:   %.loc49_12.3: %ptr.1 = bind_value %.loc49_12.2
// CHECK:STDOUT:   assign %.loc49_4.2, %.loc49_12.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
