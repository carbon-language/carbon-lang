// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
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
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.F: %Outer.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner: type = class_type @Inner [concrete]
// CHECK:STDOUT:   %ptr.36a: type = ptr_type %Inner [concrete]
// CHECK:STDOUT:   %Inner.elem.640: type = unbound_element_type %Inner, %ptr.36a [concrete]
// CHECK:STDOUT:   %ptr.5df: type = ptr_type %Outer [concrete]
// CHECK:STDOUT:   %Inner.elem.c30: type = unbound_element_type %Inner, %ptr.5df [concrete]
// CHECK:STDOUT:   %Inner.G.type: type = fn_type @Inner.G [concrete]
// CHECK:STDOUT:   %Inner.G: %Inner.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.75f: <witness> = impl_witness @Inner.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.27f: type = pattern_type %ptr.36a [concrete]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type: type = fn_type @Inner.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op: %Inner.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.pi.po.qi: type = struct_type {.pi: %ptr.36a, .po: %ptr.5df, .qi: %ptr.36a} [concrete]
// CHECK:STDOUT:   %complete_type.7ae: <witness> = complete_type_witness %struct_type.pi.po.qi [concrete]
// CHECK:STDOUT:   %Outer.H.type: type = fn_type @Outer.H [concrete]
// CHECK:STDOUT:   %Outer.H: %Outer.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.elem.a16: type = unbound_element_type %Outer, %ptr.5df [concrete]
// CHECK:STDOUT:   %Outer.elem.fe9: type = unbound_element_type %Outer, %ptr.36a [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.d28: <witness> = impl_witness @Outer.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.95c: type = pattern_type %ptr.5df [concrete]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: %Outer.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.po.qo.pi: type = struct_type {.po: %ptr.5df, .qo: %ptr.5df, .pi: %ptr.36a} [concrete]
// CHECK:STDOUT:   %complete_type.e99: <witness> = complete_type_witness %struct_type.po.qo.pi [concrete]
// CHECK:STDOUT:   %pattern_type.e74: type = pattern_type %Outer [concrete]
// CHECK:STDOUT:   %pattern_type.906: type = pattern_type %Inner [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %a.patt: %pattern_type.95c = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.95c = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.5df = value_param call_param0
// CHECK:STDOUT:     %.loc45: type = splice_block %ptr.loc45 [concrete = constants.%ptr.5df] {
// CHECK:STDOUT:       %Outer.ref.loc45: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:       %ptr.loc45: type = ptr_type %Outer.ref.loc45 [concrete = constants.%ptr.5df]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.5df = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Inner.as.Destroy.impl: @Inner.%Self.ref.loc22 as constants.%Destroy.type {
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.decl: %Inner.as.Destroy.impl.Op.type = fn_decl @Inner.as.Destroy.impl.Op [concrete = constants.%Inner.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.27f = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.27f = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc22: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.36a = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:     %self: %ptr.36a = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Inner.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Inner.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Outer.as.Destroy.impl: @Outer.%Self.ref.loc15 as constants.%Destroy.type {
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.decl: %Outer.as.Destroy.impl.Op.type = fn_decl @Outer.as.Destroy.impl.Op [concrete = constants.%Outer.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.95c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.95c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc15: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.5df = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Outer [concrete = constants.%Outer]
// CHECK:STDOUT:     %self: %ptr.5df = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Outer.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Outer.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Outer.F.decl: %Outer.F.type = fn_decl @Outer.F [concrete = constants.%Outer.F] {} {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   %Outer.H.decl: %Outer.H.type = fn_decl @Outer.H [concrete = constants.%Outer.H] {} {}
// CHECK:STDOUT:   %Self.ref.loc40: type = name_ref Self, constants.%Outer [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc40: type = ptr_type %Self.ref.loc40 [concrete = constants.%ptr.5df]
// CHECK:STDOUT:   %.loc40: %Outer.elem.a16 = field_decl po, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc41: type = ptr_type %Outer.ref [concrete = constants.%ptr.5df]
// CHECK:STDOUT:   %.loc41: %Outer.elem.a16 = field_decl qo, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc42: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   %.loc42: %Outer.elem.fe9 = field_decl pi, element2 [concrete]
// CHECK:STDOUT:   %Self.ref.loc15: type = name_ref Self, constants.%Outer [concrete = constants.%Outer]
// CHECK:STDOUT:   impl_decl @Outer.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.decl), @Outer.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.d28]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.po.qo.pi [concrete = constants.%complete_type.e99]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .F = %Outer.F.decl
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .H = %Outer.H.decl
// CHECK:STDOUT:   .po = %.loc40
// CHECK:STDOUT:   .qo = %.loc41
// CHECK:STDOUT:   .pi = %.loc42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref.loc23: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc23: type = ptr_type %Self.ref.loc23 [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   %.loc23: %Inner.elem.640 = field_decl pi, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc24: type = ptr_type %Outer.ref [concrete = constants.%ptr.5df]
// CHECK:STDOUT:   %.loc24: %Inner.elem.c30 = field_decl po, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc25: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   %.loc25: %Inner.elem.640 = field_decl qi, element2 [concrete]
// CHECK:STDOUT:   %Inner.G.decl: %Inner.G.type = fn_decl @Inner.G [concrete = constants.%Inner.G] {} {}
// CHECK:STDOUT:   %Self.ref.loc22: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:   impl_decl @Inner.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.decl), @Inner.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.75f]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.pi.po.qi [concrete = constants.%complete_type.7ae]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc23
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .po = %.loc24
// CHECK:STDOUT:   .Inner = <poisoned>
// CHECK:STDOUT:   .qi = %.loc25
// CHECK:STDOUT:   .G = %Inner.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Outer.F() {
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
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.bound: <bound method> = bound_method %i.var, constants.%Inner.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc19: %ptr.36a = addr_of %i.var
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Inner.as.Destroy.impl.Op.bound(%addr.loc19)
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.bound: <bound method> = bound_method %o.var, constants.%Outer.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc18: %ptr.5df = addr_of %o.var
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Outer.as.Destroy.impl.Op.bound(%addr.loc18)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Inner.G() {
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
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.bound: <bound method> = bound_method %i.var, constants.%Inner.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc30: %ptr.36a = addr_of %i.var
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Inner.as.Destroy.impl.Op.bound(%addr.loc30)
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.bound: <bound method> = bound_method %o.var, constants.%Outer.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc29: %ptr.5df = addr_of %o.var
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Outer.as.Destroy.impl.Op.bound(%addr.loc29)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Inner.as.Destroy.impl.Op(%self.param: %ptr.36a) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Outer.H() {
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
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.bound: <bound method> = bound_method %i.var, constants.%Inner.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc37: %ptr.36a = addr_of %i.var
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Inner.as.Destroy.impl.Op.bound(%addr.loc37)
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.bound: <bound method> = bound_method %o.var, constants.%Outer.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc36: %ptr.5df = addr_of %o.var
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Outer.as.Destroy.impl.Op.bound(%addr.loc36)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Outer.as.Destroy.impl.Op(%self.param: %ptr.5df) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %ptr.5df) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.27f = binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc46: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc46_26: ref %Outer = deref %a.ref.loc46
// CHECK:STDOUT:   %pi.ref.loc46: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc46_29.1: ref %ptr.36a = class_element_access %.loc46_26, element2
// CHECK:STDOUT:   %.loc46_21: type = splice_block %ptr.loc46 [concrete = constants.%ptr.36a] {
// CHECK:STDOUT:     %Outer.ref.loc46: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     %ptr.loc46: type = ptr_type %Inner.ref [concrete = constants.%ptr.36a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc46_29.2: %ptr.36a = bind_value %.loc46_29.1
// CHECK:STDOUT:   %b: %ptr.36a = bind_name b, %.loc46_29.2
// CHECK:STDOUT:   %a.ref.loc48_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc48_4.1: ref %Outer = deref %a.ref.loc48_3
// CHECK:STDOUT:   %po.ref.loc48: %Outer.elem.a16 = name_ref po, @Outer.%.loc40 [concrete = @Outer.%.loc40]
// CHECK:STDOUT:   %.loc48_4.2: ref %ptr.5df = class_element_access %.loc48_4.1, element0
// CHECK:STDOUT:   %a.ref.loc48_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc48_4.2, %a.ref.loc48_11
// CHECK:STDOUT:   %a.ref.loc49_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc49_4.1: ref %Outer = deref %a.ref.loc49_3
// CHECK:STDOUT:   %qo.ref: %Outer.elem.a16 = name_ref qo, @Outer.%.loc41 [concrete = @Outer.%.loc41]
// CHECK:STDOUT:   %.loc49_4.2: ref %ptr.5df = class_element_access %.loc49_4.1, element1
// CHECK:STDOUT:   %a.ref.loc49_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc49_4.2, %a.ref.loc49_11
// CHECK:STDOUT:   %a.ref.loc50_3: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc50_4.1: ref %Outer = deref %a.ref.loc50_3
// CHECK:STDOUT:   %pi.ref.loc50_4: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc50_4.2: ref %ptr.36a = class_element_access %.loc50_4.1, element2
// CHECK:STDOUT:   %a.ref.loc50_11: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc50_12.1: ref %Outer = deref %a.ref.loc50_11
// CHECK:STDOUT:   %pi.ref.loc50_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc50_12.2: ref %ptr.36a = class_element_access %.loc50_12.1, element2
// CHECK:STDOUT:   %.loc50_12.3: %ptr.36a = bind_value %.loc50_12.2
// CHECK:STDOUT:   assign %.loc50_4.2, %.loc50_12.3
// CHECK:STDOUT:   %b.ref.loc51: %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc51_4.1: ref %Inner = deref %b.ref.loc51
// CHECK:STDOUT:   %po.ref.loc51: %Inner.elem.c30 = name_ref po, @Inner.%.loc24 [concrete = @Inner.%.loc24]
// CHECK:STDOUT:   %.loc51_4.2: ref %ptr.5df = class_element_access %.loc51_4.1, element1
// CHECK:STDOUT:   %a.ref.loc51: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   assign %.loc51_4.2, %a.ref.loc51
// CHECK:STDOUT:   %b.ref.loc52: %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc52_4.1: ref %Inner = deref %b.ref.loc52
// CHECK:STDOUT:   %pi.ref.loc52_4: %Inner.elem.640 = name_ref pi, @Inner.%.loc23 [concrete = @Inner.%.loc23]
// CHECK:STDOUT:   %.loc52_4.2: ref %ptr.36a = class_element_access %.loc52_4.1, element0
// CHECK:STDOUT:   %a.ref.loc52: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc52_12.1: ref %Outer = deref %a.ref.loc52
// CHECK:STDOUT:   %pi.ref.loc52_12: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc52_12.2: ref %ptr.36a = class_element_access %.loc52_12.1, element2
// CHECK:STDOUT:   %.loc52_12.3: %ptr.36a = bind_value %.loc52_12.2
// CHECK:STDOUT:   assign %.loc52_4.2, %.loc52_12.3
// CHECK:STDOUT:   %b.ref.loc53: %ptr.36a = name_ref b, %b
// CHECK:STDOUT:   %.loc53_4.1: ref %Inner = deref %b.ref.loc53
// CHECK:STDOUT:   %qi.ref: %Inner.elem.640 = name_ref qi, @Inner.%.loc25 [concrete = @Inner.%.loc25]
// CHECK:STDOUT:   %.loc53_4.2: ref %ptr.36a = class_element_access %.loc53_4.1, element2
// CHECK:STDOUT:   %a.ref.loc53: %ptr.5df = name_ref a, %a
// CHECK:STDOUT:   %.loc53_12.1: ref %Outer = deref %a.ref.loc53
// CHECK:STDOUT:   %pi.ref.loc53: %Outer.elem.fe9 = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc53_12.2: ref %ptr.36a = class_element_access %.loc53_12.1, element2
// CHECK:STDOUT:   %.loc53_12.3: %ptr.36a = bind_value %.loc53_12.2
// CHECK:STDOUT:   assign %.loc53_4.2, %.loc53_12.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
