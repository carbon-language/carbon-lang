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
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
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
// CHECK:STDOUT:     %a.patt: %ptr.5df = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.5df = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.5df = value_param call_param0
// CHECK:STDOUT:     %.loc41: type = splice_block %ptr.loc41 [concrete = constants.%ptr.5df] {
// CHECK:STDOUT:       %Outer.ref.loc41: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:       %ptr.loc41: type = ptr_type %Outer.ref.loc41 [concrete = constants.%ptr.5df]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.5df = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %F.decl: %F.type.288 = fn_decl @F.1 [concrete = constants.%F.bd8] {} {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT:   %.loc36_9: %Outer.elem.a16 = field_decl po, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc36_3: %Outer.elem.a16 = var_pattern %.loc36_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc36: ref %Outer.elem.a16 = var <none>
// CHECK:STDOUT:   %.loc37_9: %Outer.elem.a16 = field_decl qo, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc37_3: %Outer.elem.a16 = var_pattern %.loc37_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc37: ref %Outer.elem.a16 = var <none>
// CHECK:STDOUT:   %.loc38_9: %Outer.elem.fe9 = field_decl pi, element2 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc38_3: %Outer.elem.fe9 = var_pattern %.loc38_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc38: ref %Outer.elem.fe9 = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.po.qo.pi [concrete = constants.%complete_type.e99]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .po = %.loc36_9
// CHECK:STDOUT:   .qo = %.loc37_9
// CHECK:STDOUT:   .pi = %.loc38_9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %.loc19_11: %Inner.elem.640 = field_decl pi, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc19_5: %Inner.elem.640 = var_pattern %.loc19_11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc19: ref %Inner.elem.640 = var <none>
// CHECK:STDOUT:   %.loc20_11: %Inner.elem.c30 = field_decl po, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc20_5: %Inner.elem.c30 = var_pattern %.loc20_11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc20: ref %Inner.elem.c30 = var <none>
// CHECK:STDOUT:   %.loc21_11: %Inner.elem.640 = field_decl qi, element2 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc21_5: %Inner.elem.640 = var_pattern %.loc21_11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc21: ref %Inner.elem.640 = var <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.pi.po.qi [concrete = constants.%complete_type.7ae]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc19_11
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .po = %.loc20_11
// CHECK:STDOUT:   .Inner = <poisoned>
// CHECK:STDOUT:   .qi = %.loc21_11
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %Outer = binding_pattern o
// CHECK:STDOUT:     %.loc14: %Outer = var_pattern %o.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var o
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %Inner = binding_pattern i
// CHECK:STDOUT:     %.loc15: %Inner = var_pattern %i.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var i
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %Outer = binding_pattern o
// CHECK:STDOUT:     %.loc25: %Outer = var_pattern %o.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var o
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %Inner = binding_pattern i
// CHECK:STDOUT:     %.loc26: %Inner = var_pattern %i.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var i
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %Outer = binding_pattern o
// CHECK:STDOUT:     %.loc32: %Outer = var_pattern %o.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var o
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = bind_name o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %Inner = binding_pattern i
// CHECK:STDOUT:     %.loc33: %Inner = var_pattern %i.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var i
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = bind_name i, %i.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2(%a.param_patt: %ptr.5df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %ptr.36a = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc42: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc42_26: ref %Outer = deref %a.ref.loc42
// CHECK:STDOUT:   %pi.ref.loc42: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc38_9 [concrete = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc42_29: ref %ptr.36a = class_element_access %.loc42_26, element2
// CHECK:STDOUT:   %.loc42_21: type = splice_block %ptr.loc42 [concrete = constants.%ptr.36a] {
// CHECK:STDOUT:     %Outer.ref.loc42: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     %ptr.loc42: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %ptr.36a = bind_name b, %.loc42_29
// CHECK:STDOUT:   %a.ref.loc44_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc44_4.1: ref %Outer = deref %a.ref.loc44_3
// CHECK:STDOUT:   %po.ref.loc44: %Outer.elem.a16 = name_ref po, @Outer.%.loc36_9 [concrete = @Outer.%.loc36_9]
// CHECK:STDOUT:   %.loc44_4.2: ref %ptr.5df = class_element_access %.loc44_4.1, element0
// CHECK:STDOUT:   %a.ref.loc44_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc44_4.2, %a.ref.loc44_11
// CHECK:STDOUT:   %a.ref.loc45_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc45_4.1: ref %Outer = deref %a.ref.loc45_3
// CHECK:STDOUT:   %qo.ref: %Outer.elem.a16 = name_ref qo, @Outer.%.loc37_9 [concrete = @Outer.%.loc37_9]
// CHECK:STDOUT:   %.loc45_4.2: ref %ptr.5df = class_element_access %.loc45_4.1, element1
// CHECK:STDOUT:   %a.ref.loc45_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc45_4.2, %a.ref.loc45_11
// CHECK:STDOUT:   %a.ref.loc46_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc46_4.1: ref %Outer = deref %a.ref.loc46_3
// CHECK:STDOUT:   %pi.ref.loc46_4: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc38_9 [concrete = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc46_4.2: ref %ptr.36a = class_element_access %.loc46_4.1, element2
// CHECK:STDOUT:   %a.ref.loc46_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc46_12.1: ref %Outer = deref %a.ref.loc46_11
// CHECK:STDOUT:   %pi.ref.loc46_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc38_9 [concrete = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc46_12.2: ref %ptr.36a = class_element_access %.loc46_12.1, element2
// CHECK:STDOUT:   %.loc46_12.3: %ptr.36a = bind_value %.loc46_12.2
// CHECK:STDOUT:   assign %.loc46_4.2, %.loc46_12.3
// CHECK:STDOUT:   %b.ref.loc47: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc47_3: %ptr.36a = bind_value %b.ref.loc47
// CHECK:STDOUT:   %.loc47_4.1: ref %Inner = deref %.loc47_3
// CHECK:STDOUT:   %po.ref.loc47: %Inner.elem.c30 = name_ref po, @Inner.%.loc20_11 [concrete = @Inner.%.loc20_11]
// CHECK:STDOUT:   %.loc47_4.2: ref %ptr.5df = class_element_access %.loc47_4.1, element1
// CHECK:STDOUT:   %a.ref.loc47: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc47_4.2, %a.ref.loc47
// CHECK:STDOUT:   %b.ref.loc48: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc48_3: %ptr.36a = bind_value %b.ref.loc48
// CHECK:STDOUT:   %.loc48_4.1: ref %Inner = deref %.loc48_3
// CHECK:STDOUT:   %pi.ref.loc48_4: %Inner.elem.640 = name_ref pi, @Inner.%.loc19_11 [concrete = @Inner.%.loc19_11]
// CHECK:STDOUT:   %.loc48_4.2: ref %ptr.36a = class_element_access %.loc48_4.1, element0
// CHECK:STDOUT:   %a.ref.loc48: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc48_12.1: ref %Outer = deref %a.ref.loc48
// CHECK:STDOUT:   %pi.ref.loc48_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc38_9 [concrete = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc48_12.2: ref %ptr.36a = class_element_access %.loc48_12.1, element2
// CHECK:STDOUT:   %.loc48_12.3: %ptr.36a = bind_value %.loc48_12.2
// CHECK:STDOUT:   assign %.loc48_4.2, %.loc48_12.3
// CHECK:STDOUT:   %b.ref.loc49: ref %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc49_3: %ptr.36a = bind_value %b.ref.loc49
// CHECK:STDOUT:   %.loc49_4.1: ref %Inner = deref %.loc49_3
// CHECK:STDOUT:   %qi.ref: %Inner.elem.640 = name_ref qi, @Inner.%.loc21_11 [concrete = @Inner.%.loc21_11]
// CHECK:STDOUT:   %.loc49_4.2: ref %ptr.36a = class_element_access %.loc49_4.1, element2
// CHECK:STDOUT:   %a.ref.loc49: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc49_12.1: ref %Outer = deref %a.ref.loc49
// CHECK:STDOUT:   %pi.ref.loc49: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc38_9 [concrete = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc49_12.2: ref %ptr.36a = class_element_access %.loc49_12.1, element2
// CHECK:STDOUT:   %.loc49_12.3: %ptr.36a = bind_value %.loc49_12.2
// CHECK:STDOUT:   assign %.loc49_4.2, %.loc49_12.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
