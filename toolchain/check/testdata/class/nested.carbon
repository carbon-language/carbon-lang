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
// CHECK:STDOUT:   %.1: type = ptr_type %Inner [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Inner, %.1 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %Outer [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Inner, %.3 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = struct_type {.pi: %.1, .po: %.3, .qi: %.1} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.7: type = unbound_element_type %Outer, %.3 [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type %Outer, %.1 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.po: %.3, .qo: %.3, .pi: %.1} [template]
// CHECK:STDOUT:   %.10: <witness> = complete_type_witness %.9 [template]
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
// CHECK:STDOUT:     %a.patt: %.3 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.3 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Outer.ref.loc41: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:     %.loc41: type = ptr_type %Outer [template = constants.%.3]
// CHECK:STDOUT:     %a.param: %.3 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.3 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [template = constants.%Inner] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Outer [template = constants.%Outer]
// CHECK:STDOUT:   %.loc36_15: type = ptr_type %Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc36_9: %.7 = field_decl po, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %.loc37_16: type = ptr_type %Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc37_9: %.7 = field_decl qo, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc38_16: type = ptr_type %Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc38_9: %.8 = field_decl pi, element2 [template]
// CHECK:STDOUT:   %.loc39: <witness> = complete_type_witness %.9 [template = constants.%.10]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .po = %.loc36_9
// CHECK:STDOUT:   .qo = %.loc37_9
// CHECK:STDOUT:   .pi = %.loc38_9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [template = constants.%Inner]
// CHECK:STDOUT:   %.loc19_17: type = ptr_type %Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc19_11: %.2 = field_decl pi, element0 [template]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %.loc20_18: type = ptr_type %Outer [template = constants.%.3]
// CHECK:STDOUT:   %.loc20_11: %.4 = field_decl po, element1 [template]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc21_18: type = ptr_type %Inner [template = constants.%.1]
// CHECK:STDOUT:   %.loc21_11: %.2 = field_decl qi, element2 [template]
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %.loc28: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc19_11
// CHECK:STDOUT:   .po = %.loc20_11
// CHECK:STDOUT:   .qi = %.loc21_11
// CHECK:STDOUT:   .G = %G.decl
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
// CHECK:STDOUT: fn @F.2(%a.param_patt: %.3) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref.loc42: type = name_ref Outer, file.%Outer.decl [template = constants.%Outer]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [template = constants.%Inner]
// CHECK:STDOUT:   %.loc42_21: type = ptr_type %Inner [template = constants.%.1]
// CHECK:STDOUT:   %a.ref.loc42: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc42_26: ref %Outer = deref %a.ref.loc42
// CHECK:STDOUT:   %pi.ref.loc42: %.8 = name_ref pi, @Outer.%.loc38_9 [template = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc42_29.1: ref %.1 = class_element_access %.loc42_26, element2
// CHECK:STDOUT:   %.loc42_29.2: %.1 = bind_value %.loc42_29.1
// CHECK:STDOUT:   %b: %.1 = bind_name b, %.loc42_29.2
// CHECK:STDOUT:   %a.ref.loc44_3: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc44_4.1: ref %Outer = deref %a.ref.loc44_3
// CHECK:STDOUT:   %po.ref.loc44: %.7 = name_ref po, @Outer.%.loc36_9 [template = @Outer.%.loc36_9]
// CHECK:STDOUT:   %.loc44_4.2: ref %.3 = class_element_access %.loc44_4.1, element0
// CHECK:STDOUT:   %a.ref.loc44_11: %.3 = name_ref a, %a
// CHECK:STDOUT:   assign %.loc44_4.2, %a.ref.loc44_11
// CHECK:STDOUT:   %a.ref.loc45_3: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc45_4.1: ref %Outer = deref %a.ref.loc45_3
// CHECK:STDOUT:   %qo.ref: %.7 = name_ref qo, @Outer.%.loc37_9 [template = @Outer.%.loc37_9]
// CHECK:STDOUT:   %.loc45_4.2: ref %.3 = class_element_access %.loc45_4.1, element1
// CHECK:STDOUT:   %a.ref.loc45_11: %.3 = name_ref a, %a
// CHECK:STDOUT:   assign %.loc45_4.2, %a.ref.loc45_11
// CHECK:STDOUT:   %a.ref.loc46_3: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc46_4.1: ref %Outer = deref %a.ref.loc46_3
// CHECK:STDOUT:   %pi.ref.loc46_4: %.8 = name_ref pi, @Outer.%.loc38_9 [template = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc46_4.2: ref %.1 = class_element_access %.loc46_4.1, element2
// CHECK:STDOUT:   %a.ref.loc46_11: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc46_12.1: ref %Outer = deref %a.ref.loc46_11
// CHECK:STDOUT:   %pi.ref.loc46_12: %.8 = name_ref pi, @Outer.%.loc38_9 [template = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc46_12.2: ref %.1 = class_element_access %.loc46_12.1, element2
// CHECK:STDOUT:   %.loc46_12.3: %.1 = bind_value %.loc46_12.2
// CHECK:STDOUT:   assign %.loc46_4.2, %.loc46_12.3
// CHECK:STDOUT:   %b.ref.loc47: %.1 = name_ref b, %b
// CHECK:STDOUT:   %.loc47_4.1: ref %Inner = deref %b.ref.loc47
// CHECK:STDOUT:   %po.ref.loc47: %.4 = name_ref po, @Inner.%.loc20_11 [template = @Inner.%.loc20_11]
// CHECK:STDOUT:   %.loc47_4.2: ref %.3 = class_element_access %.loc47_4.1, element1
// CHECK:STDOUT:   %a.ref.loc47: %.3 = name_ref a, %a
// CHECK:STDOUT:   assign %.loc47_4.2, %a.ref.loc47
// CHECK:STDOUT:   %b.ref.loc48: %.1 = name_ref b, %b
// CHECK:STDOUT:   %.loc48_4.1: ref %Inner = deref %b.ref.loc48
// CHECK:STDOUT:   %pi.ref.loc48_4: %.2 = name_ref pi, @Inner.%.loc19_11 [template = @Inner.%.loc19_11]
// CHECK:STDOUT:   %.loc48_4.2: ref %.1 = class_element_access %.loc48_4.1, element0
// CHECK:STDOUT:   %a.ref.loc48: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc48_12.1: ref %Outer = deref %a.ref.loc48
// CHECK:STDOUT:   %pi.ref.loc48_12: %.8 = name_ref pi, @Outer.%.loc38_9 [template = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc48_12.2: ref %.1 = class_element_access %.loc48_12.1, element2
// CHECK:STDOUT:   %.loc48_12.3: %.1 = bind_value %.loc48_12.2
// CHECK:STDOUT:   assign %.loc48_4.2, %.loc48_12.3
// CHECK:STDOUT:   %b.ref.loc49: %.1 = name_ref b, %b
// CHECK:STDOUT:   %.loc49_4.1: ref %Inner = deref %b.ref.loc49
// CHECK:STDOUT:   %qi.ref: %.2 = name_ref qi, @Inner.%.loc21_11 [template = @Inner.%.loc21_11]
// CHECK:STDOUT:   %.loc49_4.2: ref %.1 = class_element_access %.loc49_4.1, element2
// CHECK:STDOUT:   %a.ref.loc49: %.3 = name_ref a, %a
// CHECK:STDOUT:   %.loc49_12.1: ref %Outer = deref %a.ref.loc49
// CHECK:STDOUT:   %pi.ref.loc49: %.8 = name_ref pi, @Outer.%.loc38_9 [template = @Outer.%.loc38_9]
// CHECK:STDOUT:   %.loc49_12.2: ref %.1 = class_element_access %.loc49_12.1, element2
// CHECK:STDOUT:   %.loc49_12.3: %.1 = bind_value %.loc49_12.2
// CHECK:STDOUT:   assign %.loc49_4.2, %.loc49_12.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
