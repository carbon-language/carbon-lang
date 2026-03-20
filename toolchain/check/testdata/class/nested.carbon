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
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.loc15 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.01d: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.6c6) = import_ref Core//prelude/parts/default, loc34, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.ae1)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table = impl_witness_table (%Core.import_ref.01d), @T.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
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
// CHECK:STDOUT:     %.loc105: type = splice_block %ptr.loc105 [concrete = constants.%ptr.56b] {
// CHECK:STDOUT:       %Outer.ref.loc104: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:       %ptr.loc105: type = ptr_type %Outer.ref.loc104 [concrete = constants.%ptr.56b]
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
// CHECK:STDOUT:   %ptr.loc84: type = ptr_type %Self.ref [concrete = constants.%ptr.56b]
// CHECK:STDOUT:   %.loc82: %Outer.elem.1e5 = field_decl po, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc90: type = ptr_type %Outer.ref [concrete = constants.%ptr.56b]
// CHECK:STDOUT:   %.loc88: %Outer.elem.1e5 = field_decl qo, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc96: type = ptr_type %Inner.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   %.loc94: %Outer.elem.6db = field_decl pi, element2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.po.qo.pi [concrete = constants.%complete_type.c34]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Outer
// CHECK:STDOUT:   .F = %Outer.F.decl
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .H = %Outer.H.decl
// CHECK:STDOUT:   .po = %.loc82
// CHECK:STDOUT:   .qo = %.loc88
// CHECK:STDOUT:   .pi = %.loc94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Inner {
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Inner [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc29: type = ptr_type %Self.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   %.loc27: %Inner.elem.9c1 = field_decl pi, element0 [concrete]
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %ptr.loc35: type = ptr_type %Outer.ref [concrete = constants.%ptr.56b]
// CHECK:STDOUT:   %.loc33: %Inner.elem.9e0 = field_decl po, element1 [concrete]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %ptr.loc41: type = ptr_type %Inner.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   %.loc39: %Inner.elem.9c1 = field_decl qi, element2 [concrete]
// CHECK:STDOUT:   %Inner.G.decl: %Inner.G.type = fn_decl @Inner.G [concrete = constants.%Inner.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.pi.po.qi [concrete = constants.%complete_type.486]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Inner
// CHECK:STDOUT:   .pi = %.loc27
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .po = %.loc33
// CHECK:STDOUT:   .Inner = <poisoned>
// CHECK:STDOUT:   .qi = %.loc39
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
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc14: %DefaultOrUnformed.type = facet_value constants.%Outer, (constants.%DefaultOrUnformed.impl_witness.52f) [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %.loc14.1: %DefaultOrUnformed.type = converted constants.%Outer, %DefaultOrUnformed.facet.loc14 [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %as_type.loc14: type = facet_access_type %.loc14.1 [concrete = constants.%Outer]
// CHECK:STDOUT:   %.loc14.2: type = converted %.loc14.1, %as_type.loc14 [concrete = constants.%Outer]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.de4, @T.as.DefaultOrUnformed.impl.Op(constants.%Outer) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.a30]
// CHECK:STDOUT:   %.loc9: ref %Outer = splice_block %o.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc14: init %Outer to %.loc9 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %o.var, %T.as.DefaultOrUnformed.impl.Op.call.loc14
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = ref_binding o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.86a = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.86a = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc20: %DefaultOrUnformed.type = facet_value constants.%Inner, (constants.%DefaultOrUnformed.impl_witness.994) [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %.loc20.1: %DefaultOrUnformed.type = converted constants.%Inner, %DefaultOrUnformed.facet.loc20 [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %as_type.loc20: type = facet_access_type %.loc20.1 [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc20.2: type = converted %.loc20.1, %as_type.loc20 [concrete = constants.%Inner]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.da3, @T.as.DefaultOrUnformed.impl.Op(constants.%Inner) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.14b]
// CHECK:STDOUT:   %.loc15: ref %Inner = splice_block %i.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc20: init %Inner to %.loc15 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %i.var, %T.as.DefaultOrUnformed.impl.Op.call.loc20
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = ref_binding i, %i.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc15: <bound method> = bound_method %i.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc15: init %empty_tuple.type = call %Destroy.Op.bound.loc15(%i.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc9: <bound method> = bound_method %o.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc9: init %empty_tuple.type = call %Destroy.Op.bound.loc9(%o.var)
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
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc53: %DefaultOrUnformed.type = facet_value constants.%Outer, (constants.%DefaultOrUnformed.impl_witness.52f) [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %.loc53.1: %DefaultOrUnformed.type = converted constants.%Outer, %DefaultOrUnformed.facet.loc53 [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %as_type.loc53: type = facet_access_type %.loc53.1 [concrete = constants.%Outer]
// CHECK:STDOUT:   %.loc53.2: type = converted %.loc53.1, %as_type.loc53 [concrete = constants.%Outer]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.de4, @T.as.DefaultOrUnformed.impl.Op(constants.%Outer) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.a30]
// CHECK:STDOUT:   %.loc48: ref %Outer = splice_block %o.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc53: init %Outer to %.loc48 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %o.var, %T.as.DefaultOrUnformed.impl.Op.call.loc53
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = ref_binding o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.86a = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.86a = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc59: %DefaultOrUnformed.type = facet_value constants.%Inner, (constants.%DefaultOrUnformed.impl_witness.994) [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %.loc59.1: %DefaultOrUnformed.type = converted constants.%Inner, %DefaultOrUnformed.facet.loc59 [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %as_type.loc59: type = facet_access_type %.loc59.1 [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc59.2: type = converted %.loc59.1, %as_type.loc59 [concrete = constants.%Inner]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.da3, @T.as.DefaultOrUnformed.impl.Op(constants.%Inner) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.14b]
// CHECK:STDOUT:   %.loc54: ref %Inner = splice_block %i.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc59: init %Inner to %.loc54 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %i.var, %T.as.DefaultOrUnformed.impl.Op.call.loc59
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = ref_binding i, %i.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc54: <bound method> = bound_method %i.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc54: init %empty_tuple.type = call %Destroy.Op.bound.loc54(%i.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc48: <bound method> = bound_method %o.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc48: init %empty_tuple.type = call %Destroy.Op.bound.loc48(%o.var)
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
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc72: %DefaultOrUnformed.type = facet_value constants.%Outer, (constants.%DefaultOrUnformed.impl_witness.52f) [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %.loc72.1: %DefaultOrUnformed.type = converted constants.%Outer, %DefaultOrUnformed.facet.loc72 [concrete = constants.%DefaultOrUnformed.facet.83f]
// CHECK:STDOUT:   %as_type.loc72: type = facet_access_type %.loc72.1 [concrete = constants.%Outer]
// CHECK:STDOUT:   %.loc72.2: type = converted %.loc72.1, %as_type.loc72 [concrete = constants.%Outer]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.1: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.de4, @T.as.DefaultOrUnformed.impl.Op(constants.%Outer) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.a30]
// CHECK:STDOUT:   %.loc67: ref %Outer = splice_block %o.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc72: init %Outer to %.loc67 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.1()
// CHECK:STDOUT:   assign %o.var, %T.as.DefaultOrUnformed.impl.Op.call.loc72
// CHECK:STDOUT:   %Outer.ref: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:   %o: ref %Outer = ref_binding o, %o.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.86a = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.86a = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %Inner = var %i.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc78: %DefaultOrUnformed.type = facet_value constants.%Inner, (constants.%DefaultOrUnformed.impl_witness.994) [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %.loc78.1: %DefaultOrUnformed.type = converted constants.%Inner, %DefaultOrUnformed.facet.loc78 [concrete = constants.%DefaultOrUnformed.facet.6ba]
// CHECK:STDOUT:   %as_type.loc78: type = facet_access_type %.loc78.1 [concrete = constants.%Inner]
// CHECK:STDOUT:   %.loc78.2: type = converted %.loc78.1, %as_type.loc78 [concrete = constants.%Inner]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.specific_fn.2: <specific function> = specific_function constants.%T.as.DefaultOrUnformed.impl.Op.da3, @T.as.DefaultOrUnformed.impl.Op(constants.%Inner) [concrete = constants.%T.as.DefaultOrUnformed.impl.Op.specific_fn.14b]
// CHECK:STDOUT:   %.loc73: ref %Inner = splice_block %i.var {}
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call.loc78: init %Inner to %.loc73 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn.2()
// CHECK:STDOUT:   assign %i.var, %T.as.DefaultOrUnformed.impl.Op.call.loc78
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:   %i: ref %Inner = ref_binding i, %i.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc73: <bound method> = bound_method %i.var, constants.%Destroy.Op.651ba6.1
// CHECK:STDOUT:   %Destroy.Op.call.loc73: init %empty_tuple.type = call %Destroy.Op.bound.loc73(%i.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc67: <bound method> = bound_method %o.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call.loc67: init %empty_tuple.type = call %Destroy.Op.bound.loc67(%o.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15(%self.param: ref %Inner) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9(%self.param: ref %Outer) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %ptr.56b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.565 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.ref.loc118: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc117: ref %Outer = deref %a.ref.loc118
// CHECK:STDOUT:   %pi.ref.loc120: %Outer.elem.6db = name_ref pi, @Outer.%.loc94 [concrete = @Outer.%.loc94]
// CHECK:STDOUT:   %.loc120.1: ref %ptr.78a = class_element_access %.loc117, element2
// CHECK:STDOUT:   %.loc114: type = splice_block %ptr.loc114 [concrete = constants.%ptr.78a] {
// CHECK:STDOUT:     %Outer.ref.loc111: type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, @Outer.%Inner.decl [concrete = constants.%Inner]
// CHECK:STDOUT:     %ptr.loc114: type = ptr_type %Inner.ref [concrete = constants.%ptr.78a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc120.2: %ptr.78a = acquire_value %.loc120.1
// CHECK:STDOUT:   %b: %ptr.78a = value_binding b, %.loc120.2
// CHECK:STDOUT:   %a.ref.loc123: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc124.1: ref %Outer = deref %a.ref.loc123
// CHECK:STDOUT:   %po.ref.loc124: %Outer.elem.1e5 = name_ref po, @Outer.%.loc82 [concrete = @Outer.%.loc82]
// CHECK:STDOUT:   %.loc124.2: ref %ptr.56b = class_element_access %.loc124.1, element0
// CHECK:STDOUT:   %a.ref.loc127: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc127: %.be6 = impl_witness_access constants.%Copy.impl_witness.3e0, element0 [concrete = constants.%ptr.as.Copy.impl.Op.dc1]
// CHECK:STDOUT:   %bound_method.loc127.1: <bound method> = bound_method %a.ref.loc127, %impl.elem0.loc127
// CHECK:STDOUT:   %specific_fn.loc127: <specific function> = specific_function %impl.elem0.loc127, @ptr.as.Copy.impl.Op(constants.%Outer) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.904]
// CHECK:STDOUT:   %bound_method.loc127.2: <bound method> = bound_method %a.ref.loc127, %specific_fn.loc127
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc127: init %ptr.56b = call %bound_method.loc127.2(%a.ref.loc127)
// CHECK:STDOUT:   assign %.loc124.2, %ptr.as.Copy.impl.Op.call.loc127
// CHECK:STDOUT:   %a.ref.loc129: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc130.1: ref %Outer = deref %a.ref.loc129
// CHECK:STDOUT:   %qo.ref: %Outer.elem.1e5 = name_ref qo, @Outer.%.loc88 [concrete = @Outer.%.loc88]
// CHECK:STDOUT:   %.loc130.2: ref %ptr.56b = class_element_access %.loc130.1, element1
// CHECK:STDOUT:   %a.ref.loc133: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc133: %.be6 = impl_witness_access constants.%Copy.impl_witness.3e0, element0 [concrete = constants.%ptr.as.Copy.impl.Op.dc1]
// CHECK:STDOUT:   %bound_method.loc133.1: <bound method> = bound_method %a.ref.loc133, %impl.elem0.loc133
// CHECK:STDOUT:   %specific_fn.loc133: <specific function> = specific_function %impl.elem0.loc133, @ptr.as.Copy.impl.Op(constants.%Outer) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.904]
// CHECK:STDOUT:   %bound_method.loc133.2: <bound method> = bound_method %a.ref.loc133, %specific_fn.loc133
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc133: init %ptr.56b = call %bound_method.loc133.2(%a.ref.loc133)
// CHECK:STDOUT:   assign %.loc130.2, %ptr.as.Copy.impl.Op.call.loc133
// CHECK:STDOUT:   %a.ref.loc135: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc136.1: ref %Outer = deref %a.ref.loc135
// CHECK:STDOUT:   %pi.ref.loc136: %Outer.elem.6db = name_ref pi, @Outer.%.loc94 [concrete = @Outer.%.loc94]
// CHECK:STDOUT:   %.loc136.2: ref %ptr.78a = class_element_access %.loc136.1, element2
// CHECK:STDOUT:   %a.ref.loc139: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc140.1: ref %Outer = deref %a.ref.loc139
// CHECK:STDOUT:   %pi.ref.loc140: %Outer.elem.6db = name_ref pi, @Outer.%.loc94 [concrete = @Outer.%.loc94]
// CHECK:STDOUT:   %.loc140.2: ref %ptr.78a = class_element_access %.loc140.1, element2
// CHECK:STDOUT:   %.loc140.3: %ptr.78a = acquire_value %.loc140.2
// CHECK:STDOUT:   %impl.elem0.loc140: %.b9c = impl_witness_access constants.%Copy.impl_witness.044, element0 [concrete = constants.%ptr.as.Copy.impl.Op.8ab]
// CHECK:STDOUT:   %bound_method.loc140.1: <bound method> = bound_method %.loc140.3, %impl.elem0.loc140
// CHECK:STDOUT:   %specific_fn.loc140: <specific function> = specific_function %impl.elem0.loc140, @ptr.as.Copy.impl.Op(constants.%Inner) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f24]
// CHECK:STDOUT:   %bound_method.loc140.2: <bound method> = bound_method %.loc140.3, %specific_fn.loc140
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc140: init %ptr.78a = call %bound_method.loc140.2(%.loc140.3)
// CHECK:STDOUT:   assign %.loc136.2, %ptr.as.Copy.impl.Op.call.loc140
// CHECK:STDOUT:   %b.ref.loc143: %ptr.78a = name_ref b, %b
// CHECK:STDOUT:   %.loc144.1: ref %Inner = deref %b.ref.loc143
// CHECK:STDOUT:   %po.ref.loc144: %Inner.elem.9e0 = name_ref po, @Inner.%.loc33 [concrete = @Inner.%.loc33]
// CHECK:STDOUT:   %.loc144.2: ref %ptr.56b = class_element_access %.loc144.1, element1
// CHECK:STDOUT:   %a.ref.loc147: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %impl.elem0.loc147: %.be6 = impl_witness_access constants.%Copy.impl_witness.3e0, element0 [concrete = constants.%ptr.as.Copy.impl.Op.dc1]
// CHECK:STDOUT:   %bound_method.loc147.1: <bound method> = bound_method %a.ref.loc147, %impl.elem0.loc147
// CHECK:STDOUT:   %specific_fn.loc147: <specific function> = specific_function %impl.elem0.loc147, @ptr.as.Copy.impl.Op(constants.%Outer) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.904]
// CHECK:STDOUT:   %bound_method.loc147.2: <bound method> = bound_method %a.ref.loc147, %specific_fn.loc147
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc147: init %ptr.56b = call %bound_method.loc147.2(%a.ref.loc147)
// CHECK:STDOUT:   assign %.loc144.2, %ptr.as.Copy.impl.Op.call.loc147
// CHECK:STDOUT:   %b.ref.loc149: %ptr.78a = name_ref b, %b
// CHECK:STDOUT:   %.loc150.1: ref %Inner = deref %b.ref.loc149
// CHECK:STDOUT:   %pi.ref.loc150: %Inner.elem.9c1 = name_ref pi, @Inner.%.loc27 [concrete = @Inner.%.loc27]
// CHECK:STDOUT:   %.loc150.2: ref %ptr.78a = class_element_access %.loc150.1, element0
// CHECK:STDOUT:   %a.ref.loc153: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc154.1: ref %Outer = deref %a.ref.loc153
// CHECK:STDOUT:   %pi.ref.loc154: %Outer.elem.6db = name_ref pi, @Outer.%.loc94 [concrete = @Outer.%.loc94]
// CHECK:STDOUT:   %.loc154.2: ref %ptr.78a = class_element_access %.loc154.1, element2
// CHECK:STDOUT:   %.loc154.3: %ptr.78a = acquire_value %.loc154.2
// CHECK:STDOUT:   %impl.elem0.loc154: %.b9c = impl_witness_access constants.%Copy.impl_witness.044, element0 [concrete = constants.%ptr.as.Copy.impl.Op.8ab]
// CHECK:STDOUT:   %bound_method.loc154.1: <bound method> = bound_method %.loc154.3, %impl.elem0.loc154
// CHECK:STDOUT:   %specific_fn.loc154: <specific function> = specific_function %impl.elem0.loc154, @ptr.as.Copy.impl.Op(constants.%Inner) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f24]
// CHECK:STDOUT:   %bound_method.loc154.2: <bound method> = bound_method %.loc154.3, %specific_fn.loc154
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc154: init %ptr.78a = call %bound_method.loc154.2(%.loc154.3)
// CHECK:STDOUT:   assign %.loc150.2, %ptr.as.Copy.impl.Op.call.loc154
// CHECK:STDOUT:   %b.ref.loc157: %ptr.78a = name_ref b, %b
// CHECK:STDOUT:   %.loc158.1: ref %Inner = deref %b.ref.loc157
// CHECK:STDOUT:   %qi.ref: %Inner.elem.9c1 = name_ref qi, @Inner.%.loc39 [concrete = @Inner.%.loc39]
// CHECK:STDOUT:   %.loc158.2: ref %ptr.78a = class_element_access %.loc158.1, element2
// CHECK:STDOUT:   %a.ref.loc161: %ptr.56b = name_ref a, %a
// CHECK:STDOUT:   %.loc162.1: ref %Outer = deref %a.ref.loc161
// CHECK:STDOUT:   %pi.ref.loc162: %Outer.elem.6db = name_ref pi, @Outer.%.loc94 [concrete = @Outer.%.loc94]
// CHECK:STDOUT:   %.loc162.2: ref %ptr.78a = class_element_access %.loc162.1, element2
// CHECK:STDOUT:   %.loc162.3: %ptr.78a = acquire_value %.loc162.2
// CHECK:STDOUT:   %impl.elem0.loc162: %.b9c = impl_witness_access constants.%Copy.impl_witness.044, element0 [concrete = constants.%ptr.as.Copy.impl.Op.8ab]
// CHECK:STDOUT:   %bound_method.loc162.1: <bound method> = bound_method %.loc162.3, %impl.elem0.loc162
// CHECK:STDOUT:   %specific_fn.loc162: <specific function> = specific_function %impl.elem0.loc162, @ptr.as.Copy.impl.Op(constants.%Inner) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.f24]
// CHECK:STDOUT:   %bound_method.loc162.2: <bound method> = bound_method %.loc162.3, %specific_fn.loc162
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc162: init %ptr.78a = call %bound_method.loc162.2(%.loc162.3)
// CHECK:STDOUT:   assign %.loc158.2, %ptr.as.Copy.impl.Op.call.loc162
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
