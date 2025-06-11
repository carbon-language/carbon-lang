// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Outer: type = class_type @Outer [concrete]
// CHECK:STDOUT:   %F.type.288: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bd8: %F.type.288 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %ptr.36a: type = ptr_type %Inner [concrete]
// CHECK:STDOUT:   %Inner.elem.640: type = unbound_element_type %Inner, %ptr.36a [concrete]
// CHECK:STDOUT:   %ptr.5df: type = ptr_type %Outer [concrete]
// CHECK:STDOUT:   %Inner.elem.c30: type = unbound_element_type %Inner, %ptr.5df [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.pi.po.qi: type = struct_type {.pi: %ptr.36a, .po: %ptr.5df, .qi: %ptr.36a} [concrete]
// CHECK:STDOUT:   %complete_type.7ae: <witness> = complete_type_witness %struct_type.pi.po.qi [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.elem.a16: type = unbound_element_type %Outer, %ptr.5df [concrete]
// CHECK:STDOUT:   %Outer.elem.fe9: type = unbound_element_type %Outer, %ptr.36a [concrete]
// CHECK:STDOUT:   %struct_type.po.qo.pi: type = struct_type {.po: %ptr.5df, .qo: %ptr.5df, .pi: %ptr.36a} [concrete]
// CHECK:STDOUT:   %complete_type.e99: <witness> = complete_type_witness %struct_type.po.qo.pi [concrete]
// CHECK:STDOUT:   %pattern_type.e74: type = pattern_type %Outer [concrete]
// CHECK:STDOUT:   %pattern_type.906: type = pattern_type %Inner [concrete]
// CHECK:STDOUT:   %pattern_type.95c: type = pattern_type %ptr.5df [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.27f: type = pattern_type %ptr.36a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: type = class_decl @Outer [concrete = constants.%Outer] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {
// CHECK:STDOUT:     %a.patt: %pattern_type.95c = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.95c = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.5df = value_param call_param0
// CHECK:STDOUT:     %.loc44: type = splice_block %ptr.loc44 [concrete = constants.%ptr.5df] {
// CHECK:STDOUT:       %Outer.ref.loc44: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:       %ptr.loc44: type = ptr_type %Outer.ref.loc44 [concrete = constants.%ptr.5df]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.5df = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %F.decl: %F.type.288 = fn_decl @F.1 [concrete = constants.%F.bd8] {} {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Outer [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc39: type = ptr_type %Self.ref [concrete = constants.%ptr.5df]
// CHECK:STDOUT:   %.loc39: %Outer.elem.a16 = field_decl po, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc40: type = ptr_type %Outer.ref [concrete = constants.%ptr.5df]
// CHECK:STDOUT:   %.loc40: %Outer.elem.a16 = field_decl qo, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc41: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   %.loc41: %Outer.elem.fe9 = field_decl pi, element2 [concrete]
// CHECK:STDOUT:   %struct_type.po.qo.pi: type = struct_type {.po: %ptr.5df, .qo: %ptr.5df, .pi: %ptr.36a} [concrete = constants.%struct_type.po.qo.pi]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.po.qo.pi [concrete = constants.%complete_type.e99]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .po = %.loc39
// CHECK:STDOUT:   .qo = %.loc40
// CHECK:STDOUT:   .pi = %.loc41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc22: type = ptr_type %Self.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   %.loc22: %Inner.elem.640 = field_decl pi, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc23: type = ptr_type %Outer.ref [concrete = constants.%ptr.5df]
// CHECK:STDOUT:   %.loc23: %Inner.elem.c30 = field_decl po, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc24: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   %.loc24: %Inner.elem.640 = field_decl qi, element2 [concrete]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %struct_type.pi.po.qi: type = struct_type {.pi: %ptr.36a, .po: %ptr.5df, .qi: %ptr.36a} [concrete = constants.%struct_type.pi.po.qi]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.pi.po.qi [concrete = constants.%complete_type.7ae]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc22
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .po = %.loc23
// CHECK:STDOUT:   .Inner = <poisoned>
// CHECK:STDOUT:   .qi = %.loc24
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %pattern_type.e74 = binding_pattern o [concrete]
// CHECK:STDOUT:     %o.var_patt: %pattern_type.e74 = var_pattern %o.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var %o.var_patt
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.906 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.906 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %pattern_type.e74 = binding_pattern o [concrete]
// CHECK:STDOUT:     %o.var_patt: %pattern_type.e74 = var_pattern %o.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var %o.var_patt
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.906 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.906 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %pattern_type.e74 = binding_pattern o [concrete]
// CHECK:STDOUT:     %o.var_patt: %pattern_type.e74 = var_pattern %o.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var %o.var_patt
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.906 = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.906 = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param: %ptr.5df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.27f = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc45: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc45_26: ref %Outer = deref %a.ref.loc45
// CHECK:STDOUT:   %pi.ref.loc45: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc41 [concrete = @Outer.%.loc41]
// CHECK:STDOUT:   %.loc45_29: ref %ptr.36a = class_element_access %.loc45_26, element2
// CHECK:STDOUT:   %.loc45_21: type = splice_block %ptr.loc45 [concrete = constants.%ptr.36a] {
// CHECK:STDOUT:     %Outer.ref.loc45: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     %ptr.loc45: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr.36a = bind_name b, %.loc45_29
// CHECK:STDOUT:   %a.ref.loc47_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc47_4.1: ref %Outer = deref %a.ref.loc47_3
// CHECK:STDOUT:   %po.ref.loc47: %Outer.elem.a16 = name_ref po, @Outer.%.loc39 [concrete = @Outer.%.loc39]
// CHECK:STDOUT:   %.loc47_4.2: ref %ptr.5df = class_element_access %.loc47_4.1, element0
// CHECK:STDOUT:   %a.ref.loc47_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc47_4.2, %a.ref.loc47_11
// CHECK:STDOUT:   %a.ref.loc48_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc48_4.1: ref %Outer = deref %a.ref.loc48_3
// CHECK:STDOUT:   %qo.ref: %Outer.elem.a16 = name_ref qo, @Outer.%.loc40 [concrete = @Outer.%.loc40]
// CHECK:STDOUT:   %.loc48_4.2: ref %ptr.5df = class_element_access %.loc48_4.1, element1
// CHECK:STDOUT:   %a.ref.loc48_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc48_4.2, %a.ref.loc48_11
// CHECK:STDOUT:   %a.ref.loc49_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc49_4.1: ref %Outer = deref %a.ref.loc49_3
// CHECK:STDOUT:   %pi.ref.loc49_4: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc41 [concrete = @Outer.%.loc41]
// CHECK:STDOUT:   %.loc49_4.2: ref %ptr.36a = class_element_access %.loc49_4.1, element2
// CHECK:STDOUT:   %a.ref.loc49_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc49_12.1: ref %Outer = deref %a.ref.loc49_11
// CHECK:STDOUT:   %pi.ref.loc49_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc41 [concrete = @Outer.%.loc41]
// CHECK:STDOUT:   %.loc49_12.2: ref %ptr.36a = class_element_access %.loc49_12.1, element2
// CHECK:STDOUT:   %.loc49_12.3: %ptr.36a = bind_value %.loc49_12.2
// CHECK:STDOUT:   assign %.loc49_4.2, %.loc49_12.3
// CHECK:STDOUT:   %b.ref.loc50: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc50_3: %ptr.36a = bind_value %b.ref.loc50
// CHECK:STDOUT:   %.loc50_4.1: ref %Inner = deref %.loc50_3
// CHECK:STDOUT:   %po.ref.loc50: %Inner.elem.c30 = name_ref po, @Inner.%.loc23 [concrete = @Inner.%.loc23]
// CHECK:STDOUT:   %.loc50_4.2: ref %ptr.5df = class_element_access %.loc50_4.1, element1
// CHECK:STDOUT:   %a.ref.loc50: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc50_4.2, %a.ref.loc50
// CHECK:STDOUT:   %b.ref.loc51: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc51_3: %ptr.36a = bind_value %b.ref.loc51
// CHECK:STDOUT:   %.loc51_4.1: ref %Inner = deref %.loc51_3
// CHECK:STDOUT:   %pi.ref.loc51_4: %Inner.elem.640 = name_ref pi, @Inner.%.loc22 [concrete = @Inner.%.loc22]
// CHECK:STDOUT:   %.loc51_4.2: ref %ptr.36a = class_element_access %.loc51_4.1, element0
// CHECK:STDOUT:   %a.ref.loc51: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc51_12.1: ref %Outer = deref %a.ref.loc51
// CHECK:STDOUT:   %pi.ref.loc51_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc41 [concrete = @Outer.%.loc41]
// CHECK:STDOUT:   %.loc51_12.2: ref %ptr.36a = class_element_access %.loc51_12.1, element2
// CHECK:STDOUT:   %.loc51_12.3: %ptr.36a = bind_value %.loc51_12.2
// CHECK:STDOUT:   assign %.loc51_4.2, %.loc51_12.3
// CHECK:STDOUT:   %b.ref.loc52: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc52_3: %ptr.36a = bind_value %b.ref.loc52
// CHECK:STDOUT:   %.loc52_4.1: ref %Inner = deref %.loc52_3
// CHECK:STDOUT:   %qi.ref: %Inner.elem.640 = name_ref qi, @Inner.%.loc24 [concrete = @Inner.%.loc24]
// CHECK:STDOUT:   %.loc52_4.2: ref %ptr.36a = class_element_access %.loc52_4.1, element2
// CHECK:STDOUT:   %a.ref.loc52: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc52_12.1: ref %Outer = deref %a.ref.loc52
// CHECK:STDOUT:   %pi.ref.loc52: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc41 [concrete = @Outer.%.loc41]
// CHECK:STDOUT:   %.loc52_12.2: ref %ptr.36a = class_element_access %.loc52_12.1, element2
// CHECK:STDOUT:   %.loc52_12.3: %ptr.36a = bind_value %.loc52_12.2
// CHECK:STDOUT:   assign %.loc52_4.2, %.loc52_12.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
