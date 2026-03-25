// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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
    var unused o: Outer;
    var unused i: Inner;
  }

  class Inner {
    var pi: Self*;
    var po: Outer*;
    var qi: Inner*;

    fn G() {
      // Outer and Inner are both complete here.
      var unused o: Outer;
      var unused i: Inner;
    }
  }

  fn H() {
    // Outer and Inner are both complete here.
    var unused o: Outer;
    var unused i: Inner;
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
// CHECK:STDOUT:   %ptr.78a: type = ptr_type %Inner [concrete]
// CHECK:STDOUT:   %Inner.elem.9c1: type = unbound_element_type %Inner, %ptr.78a [concrete]
// CHECK:STDOUT:   %ptr.56b: type = ptr_type %Outer [concrete]
// CHECK:STDOUT:   %Inner.elem.9e0: type = unbound_element_type %Inner, %ptr.56b [concrete]
// CHECK:STDOUT:   %Inner.G.type: type = fn_type @Inner.G [concrete]
// CHECK:STDOUT:   %Inner.G: %Inner.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.pi.po.qi: type = struct_type {.pi: %ptr.78a, .po: %ptr.56b, .qi: %ptr.78a} [concrete]
// CHECK:STDOUT:   %complete_type.486: <witness> = complete_type_witness %struct_type.pi.po.qi [concrete]
// CHECK:STDOUT:   %Outer.H.type: type = fn_type @Outer.H [concrete]
// CHECK:STDOUT:   %Outer.H: %Outer.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.elem.1e5: type = unbound_element_type %Outer, %ptr.56b [concrete]
// CHECK:STDOUT:   %Outer.elem.6db: type = unbound_element_type %Outer, %ptr.78a [concrete]
// CHECK:STDOUT:   %struct_type.po.qo.pi: type = struct_type {.po: %ptr.56b, .qo: %ptr.56b, .pi: %ptr.78a} [concrete]
// CHECK:STDOUT:   %complete_type.c34: <witness> = complete_type_witness %struct_type.po.qo.pi [concrete]
// CHECK:STDOUT:   %pattern_type.9ae: type = pattern_type %Outer [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.6c6: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.ae1: %T.as.DefaultOrUnformed.impl.Op.type.6c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.52f: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%Outer) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.8b8: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Outer) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.de4: %T.as.DefaultOrUnformed.impl.Op.type.8b8 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.83f: %DefaultOrUnformed.type = facet_value %Outer, (%DefaultOrUnformed.impl_witness.52f) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.a30: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.de4, @T.as.DefaultOrUnformed.impl.Op(%Outer) [concrete]
// CHECK:STDOUT:   %pattern_type.86a: type = pattern_type %Inner [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.994: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table, @T.as.DefaultOrUnformed.impl(%Inner) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.036: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%Inner) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.da3: %T.as.DefaultOrUnformed.impl.Op.type.036 = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.6ba: %DefaultOrUnformed.type = facet_value %Inner, (%DefaultOrUnformed.impl_witness.994) [concrete]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.14b: <specific function> = specific_function %T.as.DefaultOrUnformed.impl.Op.da3, @T.as.DefaultOrUnformed.impl.Op(%Inner) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc19 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc18 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cd9: type = pattern_type %ptr.56b [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.565: type = pattern_type %ptr.78a [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.3e0: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%Outer) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.561: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Outer) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.dc1: %ptr.as.Copy.impl.Op.type.561 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.e2d: %Copy.type = facet_value %ptr.56b, (%Copy.impl_witness.3e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ed5: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.e2d) [concrete]
// CHECK:STDOUT:   %.be6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ed5, %Copy.facet.e2d [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.904: <specific function> = specific_function %ptr.as.Copy.impl.Op.dc1, @ptr.as.Copy.impl.Op(%Outer) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.044: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%Inner) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.72d: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Inner) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8ab: %ptr.as.Copy.impl.Op.type.72d = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.ea8: %Copy.type = facet_value %ptr.78a, (%Copy.impl_witness.044) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.4b9: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ea8) [concrete]
// CHECK:STDOUT:   %.b9c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.4b9, %Copy.facet.ea8 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.f24: <specific function> = specific_function %ptr.as.Copy.impl.Op.8ab, @ptr.as.Copy.impl.Op(%Inner) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.cd9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.cd9 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.56b = value_param call_param0
// CHECK:STDOUT:     %.loc45: type = splice_block %ptr.loc45 [concrete = constants.%ptr.56b] {
// CHECK:STDOUT:       %Outer.ref.loc45: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:       %ptr.loc45: type = ptr_type %Outer.ref.loc45 [concrete = constants.%ptr.56b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.56b = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Outer {
// CHECK:STDOUT:   %Outer.F.decl: %Outer.F.type = fn_decl @Outer.F [concrete = constants.%Outer.F] {} {}
// CHECK:STDOUT:   %Inner.decl: type = class_decl @Inner [concrete = constants.%Inner] {} {}
// CHECK:STDOUT:   %Outer.H.decl: %Outer.H.type = fn_decl @Outer.H [concrete = constants.%Outer.H] {} {}
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Outer [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc40: type = ptr_type %Self.ref [concrete = constants.%ptr.56b]
// CHECK:STDOUT:   %.loc40: %Outer.elem.1e5 = field_decl po, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc41: type = ptr_type %Outer.ref [concrete = constants.%ptr.56b]
// CHECK:STDOUT:   %.loc41: %Outer.elem.1e5 = field_decl qo, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc42: type = ptr_type %Inner.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   %.loc42: %Outer.elem.6db = field_decl pi, element2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.po.qo.pi [concrete = constants.%complete_type.c34]
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc23: type = ptr_type %Self.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   %.loc23: %Inner.elem.9c1 = field_decl pi, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc24: type = ptr_type %Outer.ref [concrete = constants.%ptr.56b]
// CHECK:STDOUT:   %.loc24: %Inner.elem.9e0 = field_decl po, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc25: type = ptr_type %Inner.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   %.loc25: %Inner.elem.9c1 = field_decl qi, element2 [concrete]
// CHECK:STDOUT:   %Inner.G.decl: %Inner.G.type = fn_decl @Inner.G [concrete = constants.%Inner.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.pi.po.qi [concrete = constants.%complete_type.486]
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
// CHECK:STDOUT:     %o.patt: %pattern_type.9ae = ref_binding_pattern o [concrete]
// CHECK:STDOUT:     %o.var_patt: %pattern_type.9ae = var_pattern %o.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var %o.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc18: %DefaultOrUnformed.type = facet_value constants.%Outer, (constants.%DefaultOrUnformed.impl_witness.52f) [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %.loc18_24.1: %DefaultOrUnformed.type = converted constants.%Outer, %DefaultOrUnformed.facet.loc18 [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %as_type.loc18: type = facet_access_type %.loc18_24.1 [concrete = constants.%Outer]
// CHECK:STDOUT:   %.loc18_24.2: type = converted %.loc18_24.1, %as_type.loc18 [concrete = constants.%Outer]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.de4, @T.as.DefaultOrUnformed.impl.Op(constants.%Outer) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.a30]
// CHECK:STDOUT:   %.loc18_5: ref %Outer = splice_block %o.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc18: init %Outer to %.loc18_5 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %o.var, %T.as.DefaultOrUnformed.impl.Op.call.loc18
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = ref_binding o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.86a = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.86a = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc19: %DefaultOrUnformed.type = facet_value constants.%Inner, (constants.%DefaultOrUnformed.impl_witness.994) [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %.loc19_24.1: %DefaultOrUnformed.type = converted constants.%Inner, %DefaultOrUnformed.facet.loc19 [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %as_type.loc19: type = facet_access_type %.loc19_24.1 [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc19_24.2: type = converted %.loc19_24.1, %as_type.loc19 [concrete = constants.%Inner]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.da3, @T.as.DefaultOrUnformed.impl.Op(constants.%Inner) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.14b]
// CHECK:STDOUT:   %.loc19_5: ref %Inner = splice_block %i.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc19: init %Inner to %.loc19_5 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %i.var, %T.as.DefaultOrUnformed.impl.Op.call.loc19
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = ref_binding i, %i.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc19: <bound method> = bound_method %i.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc19: init %empty_tuple.type = call %Destroy.Op.bound.loc19(%i.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc18: <bound method> = bound_method %o.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc18: init %empty_tuple.type = call %Destroy.Op.bound.loc18(%o.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Inner.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %pattern_type.9ae = ref_binding_pattern o [concrete]
// CHECK:STDOUT:     %o.var_patt: %pattern_type.9ae = var_pattern %o.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var %o.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc29: %DefaultOrUnformed.type = facet_value constants.%Outer, (constants.%DefaultOrUnformed.impl_witness.52f) [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %.loc29_26.1: %DefaultOrUnformed.type = converted constants.%Outer, %DefaultOrUnformed.facet.loc29 [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %as_type.loc29: type = facet_access_type %.loc29_26.1 [concrete = constants.%Outer]
// CHECK:STDOUT:   %.loc29_26.2: type = converted %.loc29_26.1, %as_type.loc29 [concrete = constants.%Outer]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.de4, @T.as.DefaultOrUnformed.impl.Op(constants.%Outer) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.a30]
// CHECK:STDOUT:   %.loc29_7: ref %Outer = splice_block %o.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc29: init %Outer to %.loc29_7 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %o.var, %T.as.DefaultOrUnformed.impl.Op.call.loc29
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = ref_binding o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.86a = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.86a = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc30: %DefaultOrUnformed.type = facet_value constants.%Inner, (constants.%DefaultOrUnformed.impl_witness.994) [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %.loc30_26.1: %DefaultOrUnformed.type = converted constants.%Inner, %DefaultOrUnformed.facet.loc30 [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %as_type.loc30: type = facet_access_type %.loc30_26.1 [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc30_26.2: type = converted %.loc30_26.1, %as_type.loc30 [concrete = constants.%Inner]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.da3, @T.as.DefaultOrUnformed.impl.Op(constants.%Inner) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.14b]
// CHECK:STDOUT:   %.loc30_7: ref %Inner = splice_block %i.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc30: init %Inner to %.loc30_7 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %i.var, %T.as.DefaultOrUnformed.impl.Op.call.loc30
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = ref_binding i, %i.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc30: <bound method> = bound_method %i.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc30: init %empty_tuple.type = call %Destroy.Op.bound.loc30(%i.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc29: <bound method> = bound_method %o.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc29: init %empty_tuple.type = call %Destroy.Op.bound.loc29(%o.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Outer.H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %o.patt: %pattern_type.9ae = ref_binding_pattern o [concrete]
// CHECK:STDOUT:     %o.var_patt: %pattern_type.9ae = var_pattern %o.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %o.var: ref %Outer = var %o.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc36: %DefaultOrUnformed.type = facet_value constants.%Outer, (constants.%DefaultOrUnformed.impl_witness.52f) [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %.loc36_24.1: %DefaultOrUnformed.type = converted constants.%Outer, %DefaultOrUnformed.facet.loc36 [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %as_type.loc36: type = facet_access_type %.loc36_24.1 [concrete = constants.%Outer]
// CHECK:STDOUT:   %.loc36_24.2: type = converted %.loc36_24.1, %as_type.loc36 [concrete = constants.%Outer]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.de4, @T.as.DefaultOrUnformed.impl.Op(constants.%Outer) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.a30]
// CHECK:STDOUT:   %.loc36_5: ref %Outer = splice_block %o.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc36: init %Outer to %.loc36_5 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %o.var, %T.as.DefaultOrUnformed.impl.Op.call.loc36
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = ref_binding o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.86a = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.86a = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc37: %DefaultOrUnformed.type = facet_value constants.%Inner, (constants.%DefaultOrUnformed.impl_witness.994) [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %.loc37_24.1: %DefaultOrUnformed.type = converted constants.%Inner, %DefaultOrUnformed.facet.loc37 [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %as_type.loc37: type = facet_access_type %.loc37_24.1 [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc37_24.2: type = converted %.loc37_24.1, %as_type.loc37 [concrete = constants.%Inner]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.da3, @T.as.DefaultOrUnformed.impl.Op(constants.%Inner) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.14b]
// CHECK:STDOUT:   %.loc37_5: ref %Inner = splice_block %i.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc37: init %Inner to %.loc37_5 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %i.var, %T.as.DefaultOrUnformed.impl.Op.call.loc37
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = ref_binding i, %i.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc37: <bound method> = bound_method %i.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc37: init %empty_tuple.type = call %Destroy.Op.bound.loc37(%i.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc36: <bound method> = bound_method %o.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc36: init %empty_tuple.type = call %Destroy.Op.bound.loc36(%o.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc19(%self.param: ref %Inner) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc18(%self.param: ref %Outer) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %ptr.56b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.565 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc46: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc46_26: ref %Outer = deref %a.ref.loc46
// CHECK:STDOUT:   %pi.ref.loc46: %Outer.elem.6db = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc46_29.1: ref %ptr.78a = class_element_access %.loc46_26, element2
// CHECK:STDOUT:   %.loc46_21: type = splice_block %ptr.loc46 [concrete = constants.%ptr.78a] {
// CHECK:STDOUT:     %Outer.ref.loc46: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     %ptr.loc46: type = ptr_type %Inner.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc46_29.2: %ptr.78a = acquire_value %.loc46_29.1
// CHECK:STDOUT:   %b: %ptr.78a = value_binding b, %.loc46_29.2
// CHECK:STDOUT:   %a.ref.loc48_3: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc48_4.1: ref %Outer = deref %a.ref.loc48_3
// CHECK:STDOUT:   %po.ref.loc48: %Outer.elem.1e5 = name_ref po, @Outer.%.loc40 [concrete = @Outer.%.loc40]
// CHECK:STDOUT:   %.loc48_4.2: ref %ptr.56b = class_element_access %.loc48_4.1, element0
// CHECK:STDOUT:   %a.ref.loc48_11: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc48: %.be6 = impl_witness_access constants.%Copy.impl_witness.3e0, element0 [concrete = constants.%ptr.as.Copy.impl.Op.dc1]
// CHECK:STDOUT:   %bound_method.loc48_11.1: <bound method> = bound_method %a.ref.loc48_11, %impl.elem0.loc48
// CHECK:STDOUT:   %specific_fn.loc48: <specific function> = specific_function %impl.elem0.loc48, @ptr.as.Copy.impl.Op(constants.%Outer) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.904]
// CHECK:STDOUT:   %bound_method.loc48_11.2: <bound method> = bound_method %a.ref.loc48_11, %specific_fn.loc48
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc48: init %ptr.56b = call %bound_method.loc48_11.2(%a.ref.loc48_11)
// CHECK:STDOUT:   assign %.loc48_4.2, %ptr.as.Copy.impl.Op.call.loc48
// CHECK:STDOUT:   %a.ref.loc49_3: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc49_4.1: ref %Outer = deref %a.ref.loc49_3
// CHECK:STDOUT:   %qo.ref: %Outer.elem.1e5 = name_ref qo, @Outer.%.loc41 [concrete = @Outer.%.loc41]
// CHECK:STDOUT:   %.loc49_4.2: ref %ptr.56b = class_element_access %.loc49_4.1, element1
// CHECK:STDOUT:   %a.ref.loc49_11: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc49: %.be6 = impl_witness_access constants.%Copy.impl_witness.3e0, element0 [concrete = constants.%ptr.as.Copy.impl.Op.dc1]
// CHECK:STDOUT:   %bound_method.loc49_11.1: <bound method> = bound_method %a.ref.loc49_11, %impl.elem0.loc49
// CHECK:STDOUT:   %specific_fn.loc49: <specific function> = specific_function %impl.elem0.loc49, @ptr.as.Copy.impl.Op(constants.%Outer) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.904]
// CHECK:STDOUT:   %bound_method.loc49_11.2: <bound method> = bound_method %a.ref.loc49_11, %specific_fn.loc49
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc49: init %ptr.56b = call %bound_method.loc49_11.2(%a.ref.loc49_11)
// CHECK:STDOUT:   assign %.loc49_4.2, %ptr.as.Copy.impl.Op.call.loc49
// CHECK:STDOUT:   %a.ref.loc50_3: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc50_4.1: ref %Outer = deref %a.ref.loc50_3
// CHECK:STDOUT:   %pi.ref.loc50_4: %Outer.elem.6db = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc50_4.2: ref %ptr.78a = class_element_access %.loc50_4.1, element2
// CHECK:STDOUT:   %a.ref.loc50_11: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc50_12.1: ref %Outer = deref %a.ref.loc50_11
// CHECK:STDOUT:   %pi.ref.loc50_12: %Outer.elem.6db = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc50_12.2: ref %ptr.78a = class_element_access %.loc50_12.1, element2
// CHECK:STDOUT:   %.loc50_12.3: %ptr.78a = acquire_value %.loc50_12.2
// CHECK:STDOUT:   %impl.elem0.loc50: %.b9c = impl_witness_access constants.%Copy.impl_witness.044, element0 [concrete = constants.%ptr.as.Copy.impl.Op.8ab]
// CHECK:STDOUT:   %bound_method.loc50_12.1: <bound method> = bound_method %.loc50_12.3, %impl.elem0.loc50
// CHECK:STDOUT:   %specific_fn.loc50: <specific function> = specific_function %impl.elem0.loc50, @ptr.as.Copy.impl.Op(constants.%Inner) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f24]
// CHECK:STDOUT:   %bound_method.loc50_12.2: <bound method> = bound_method %.loc50_12.3, %specific_fn.loc50
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc50: init %ptr.78a = call %bound_method.loc50_12.2(%.loc50_12.3)
// CHECK:STDOUT:   assign %.loc50_4.2, %ptr.as.Copy.impl.Op.call.loc50
// CHECK:STDOUT:   %b.ref.loc51: %ptr.78a = name_ref b, %b
// CHECK:STDOUT:   %.loc51_4.1: ref %Inner = deref %b.ref.loc51
// CHECK:STDOUT:   %po.ref.loc51: %Inner.elem.9e0 = name_ref po, @Inner.%.loc24 [concrete = @Inner.%.loc24]
// CHECK:STDOUT:   %.loc51_4.2: ref %ptr.56b = class_element_access %.loc51_4.1, element1
// CHECK:STDOUT:   %a.ref.loc51: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc51: %.be6 = impl_witness_access constants.%Copy.impl_witness.3e0, element0 [concrete = constants.%ptr.as.Copy.impl.Op.dc1]
// CHECK:STDOUT:   %bound_method.loc51_11.1: <bound method> = bound_method %a.ref.loc51, %impl.elem0.loc51
// CHECK:STDOUT:   %specific_fn.loc51: <specific function> = specific_function %impl.elem0.loc51, @ptr.as.Copy.impl.Op(constants.%Outer) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.904]
// CHECK:STDOUT:   %bound_method.loc51_11.2: <bound method> = bound_method %a.ref.loc51, %specific_fn.loc51
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc51: init %ptr.56b = call %bound_method.loc51_11.2(%a.ref.loc51)
// CHECK:STDOUT:   assign %.loc51_4.2, %ptr.as.Copy.impl.Op.call.loc51
// CHECK:STDOUT:   %b.ref.loc52: %ptr.78a = name_ref b, %b
// CHECK:STDOUT:   %.loc52_4.1: ref %Inner = deref %b.ref.loc52
// CHECK:STDOUT:   %pi.ref.loc52_4: %Inner.elem.9c1 = name_ref pi, @Inner.%.loc23 [concrete = @Inner.%.loc23]
// CHECK:STDOUT:   %.loc52_4.2: ref %ptr.78a = class_element_access %.loc52_4.1, element0
// CHECK:STDOUT:   %a.ref.loc52: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc52_12.1: ref %Outer = deref %a.ref.loc52
// CHECK:STDOUT:   %pi.ref.loc52_12: %Outer.elem.6db = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc52_12.2: ref %ptr.78a = class_element_access %.loc52_12.1, element2
// CHECK:STDOUT:   %.loc52_12.3: %ptr.78a = acquire_value %.loc52_12.2
// CHECK:STDOUT:   %impl.elem0.loc52: %.b9c = impl_witness_access constants.%Copy.impl_witness.044, element0 [concrete = constants.%ptr.as.Copy.impl.Op.8ab]
// CHECK:STDOUT:   %bound_method.loc52_12.1: <bound method> = bound_method %.loc52_12.3, %impl.elem0.loc52
// CHECK:STDOUT:   %specific_fn.loc52: <specific function> = specific_function %impl.elem0.loc52, @ptr.as.Copy.impl.Op(constants.%Inner) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f24]
// CHECK:STDOUT:   %bound_method.loc52_12.2: <bound method> = bound_method %.loc52_12.3, %specific_fn.loc52
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc52: init %ptr.78a = call %bound_method.loc52_12.2(%.loc52_12.3)
// CHECK:STDOUT:   assign %.loc52_4.2, %ptr.as.Copy.impl.Op.call.loc52
// CHECK:STDOUT:   %b.ref.loc53: %ptr.78a = name_ref b, %b
// CHECK:STDOUT:   %.loc53_4.1: ref %Inner = deref %b.ref.loc53
// CHECK:STDOUT:   %qi.ref: %Inner.elem.9c1 = name_ref qi, @Inner.%.loc25 [concrete = @Inner.%.loc25]
// CHECK:STDOUT:   %.loc53_4.2: ref %ptr.78a = class_element_access %.loc53_4.1, element2
// CHECK:STDOUT:   %a.ref.loc53: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc53_12.1: ref %Outer = deref %a.ref.loc53
// CHECK:STDOUT:   %pi.ref.loc53: %Outer.elem.6db = name_ref pi, @Outer.%.loc42 [concrete = @Outer.%.loc42]
// CHECK:STDOUT:   %.loc53_12.2: ref %ptr.78a = class_element_access %.loc53_12.1, element2
// CHECK:STDOUT:   %.loc53_12.3: %ptr.78a = acquire_value %.loc53_12.2
// CHECK:STDOUT:   %impl.elem0.loc53: %.b9c = impl_witness_access constants.%Copy.impl_witness.044, element0 [concrete = constants.%ptr.as.Copy.impl.Op.8ab]
// CHECK:STDOUT:   %bound_method.loc53_12.1: <bound method> = bound_method %.loc53_12.3, %impl.elem0.loc53
// CHECK:STDOUT:   %specific_fn.loc53: <specific function> = specific_function %impl.elem0.loc53, @ptr.as.Copy.impl.Op(constants.%Inner) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f24]
// CHECK:STDOUT:   %bound_method.loc53_12.2: <bound method> = bound_method %.loc53_12.3, %specific_fn.loc53
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc53: init %ptr.78a = call %bound_method.loc53_12.2(%.loc53_12.3)
// CHECK:STDOUT:   assign %.loc53_4.2, %ptr.as.Copy.impl.Op.call.loc53
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
