// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_type.carbon

// --- class.carbon

library "[[@TEST_NAME]]";

class Outer(T:! Core.Copy) {
  class Inner {
    var n: T;
  }

  fn F(n: T) -> Inner { return {.n = n}; }
}

fn Test() -> i32 {
  var c: Outer(i32).Inner = Outer(i32).F(1);
  return c.n;
}

// --- interface.carbon

library "[[@TEST_NAME]]";

class Outer(T:! type) {
  interface Inner {
    fn F[self: Self]() -> T;
  }

  class C {
    impl as Inner {
      fn F[self: C]() -> T { return self.(Inner.F)(); }
    }
  }
}

class D {
  impl as Outer(i32).Inner {
    fn F[self: D]() -> i32;
  }
}

fn Test() -> i32 {
  var c: Outer(i32).C = {};
  return c.(Outer(i32).Inner.F)();
}

// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.35b: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.228: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.6d8: type = class_type @Outer, @Outer(%T.228) [symbolic]
// CHECK:STDOUT:   %Inner.772: type = class_type @Inner, @Inner(%T.228) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.228 [symbolic]
// CHECK:STDOUT:   %require_complete.c28: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.4b69d0.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.elem.b32: type = unbound_element_type %Inner.772, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n.5e5: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.795: <witness> = complete_type_witness %struct_type.n.5e5 [symbolic]
// CHECK:STDOUT:   %.293: Core.Form = init_form %Inner.772 [symbolic]
// CHECK:STDOUT:   %pattern_type.c64: type = pattern_type %Inner.772 [symbolic]
// CHECK:STDOUT:   %Outer.F.type.f80: type = fn_type @Outer.F, @Outer(%T.228) [symbolic]
// CHECK:STDOUT:   %Outer.F.3e5: %Outer.F.type.f80 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.000: <witness> = require_complete_type %Inner.772 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c49: <witness> = lookup_impl_witness %T.228, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2ec0c.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.228) [symbolic]
// CHECK:STDOUT:   %.12f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2ec0c.2, %T.228 [symbolic]
// CHECK:STDOUT:   %impl.elem0.cbd: %.12f = impl_witness_access %Copy.lookup_impl_witness.c49, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.c5f: <specific function> = specific_impl_function %impl.elem0.cbd, @Copy.WithSelf.Op(%T.228) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.66a: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Outer.76a: type = class_type @Outer, @Outer(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %Inner.4b6: type = class_type @Inner, @Inner(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %Outer.F.type.34a: type = fn_type @Outer.F, @Outer(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %Outer.F.8c7: %Outer.F.type.34a = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.e1c: type = unbound_element_type %Inner.4b6, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.179: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.059: <witness> = complete_type_witness %struct_type.n.179 [concrete]
// CHECK:STDOUT:   %pattern_type.f69: type = pattern_type %Inner.4b6 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.470: <witness> = impl_witness imports.%Copy.impl_witness_table.73d [concrete]
// CHECK:STDOUT:   %Copy.facet.fe3: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.470) [concrete]
// CHECK:STDOUT:   %.6f1: Core.Form = init_form %Inner.4b6 [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.8c7, @Outer.F(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d5e: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet.66a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc13_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.2da = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.73d = impl_witness_table (%Core.import_ref.2da), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.35b = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.228)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc12 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32.loc12 [concrete = constants.%.70c]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc4_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.228)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_14.1) [symbolic = %Inner (constants.%Inner.772)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_14.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.f80)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.f80) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.3e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.772)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.f80) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.3e5)] {
// CHECK:STDOUT:       %n.param_patt: @Outer.F.%pattern_type.loc9_9 (%pattern_type.4b69d0.2) = value_param_pattern [concrete]
// CHECK:STDOUT:       %n.patt: @Outer.F.%pattern_type.loc9_9 (%pattern_type.4b69d0.2) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Outer.F.%pattern_type.loc9_17 (%pattern_type.c64) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Outer.F.%pattern_type.loc9_17 (%pattern_type.c64) = return_slot_pattern %return.param_patt, %Inner.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.228) [symbolic = %Inner (constants.%Inner.772)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17.2 [symbolic = %Inner (constants.%Inner.772)]
// CHECK:STDOUT:       %.loc9_17.3: Core.Form = init_form %Inner.ref [symbolic = %.loc9_17.1 (constants.%.293)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T.228)]
// CHECK:STDOUT:         %T.as_type.loc9_11.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc9_11.2: type = converted %T.ref, %T.as_type.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.772) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.772) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.6d8
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.228)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.c28)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.772)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.as_type [symbolic = %Inner.elem (constants.%Inner.elem.b32)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.as_type (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.5e5)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.795)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6: @Inner.%Inner.elem (%Inner.elem.b32) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.5e5 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.795)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.772
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.228)]
// CHECK:STDOUT:   %T.as_type.loc9_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_9: type = pattern_type %T.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_9 (constants.%pattern_type.4b69d0.2)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.772)]
// CHECK:STDOUT:   %.loc9_17.1: Core.Form = init_form %Inner [symbolic = %.loc9_17.1 (constants.%.293)]
// CHECK:STDOUT:   %pattern_type.loc9_17: type = pattern_type %Inner [symbolic = %pattern_type.loc9_17 (constants.%pattern_type.c64)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.as_type.loc9_11.1 [symbolic = %require_complete.loc9_9 (constants.%require_complete.c28)]
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.000)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.5e5)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2ec0c.2)]
// CHECK:STDOUT:   %.loc9_38: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_38 (constants.%.12f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c49)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38 (%.12f) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)) -> out %return.param: @Outer.F.%Inner (%Inner.772) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.5e5) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38 (%.12f) = impl_witness_access constants.%Copy.lookup_impl_witness.c49, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.cbd)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.WithSelf.Op(constants.%T.228) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.c5f)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = class_element_access %return.param, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = call %bound_method.loc9_38.2(%n.ref)
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.772) to %return.param = class_init (%.loc9_39.3)
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.772) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.f69 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.f69 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.4b6 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc13_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value %i32.loc13_35, (constants.%Copy.impl_witness.32d) [concrete = constants.%Copy.facet.66a]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.66a]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.66a) [concrete = constants.%Outer.76a]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.34a = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.66a) [concrete = constants.%Outer.F.8c7]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.34a = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.8c7]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.470) [concrete = constants.%Copy.facet.fe3]
// CHECK:STDOUT:   %.loc13_43: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43 [concrete = constants.%Copy.facet.fe3]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.66a) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3: ref %Inner.4b6 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0.loc13: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.d5e]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.4b6 to %.loc13_3 = call %Outer.F.specific_fn(%.loc13_42.2)
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.4b6] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc13_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value %i32.loc13_16, (constants.%Copy.impl_witness.32d) [concrete = constants.%Copy.facet.66a]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.66a]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.66a) [concrete = constants.%Outer.76a]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.66a) [concrete = constants.%Inner.4b6]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.4b6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.4b6 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.4b6 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.e1c = name_ref n, @Inner.%.loc6 [concrete = @Inner.%.loc6]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = acquire_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0.loc14: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn.loc14
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%c.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.3(%self.param: ref %struct_type.n.179) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.4(%self.param: ref %Inner.4b6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.228) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.228
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.772
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.f80
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.3e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.228) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.228
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.c28
// CHECK:STDOUT:   %Inner => constants.%Inner.772
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.b32
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.5e5
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.795
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.228) {
// CHECK:STDOUT:   %T => constants.%T.228
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.4b69d0.2
// CHECK:STDOUT:   %Inner => constants.%Inner.772
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.293
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.c64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.66a) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.66a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.4b6
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.34a
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.8c7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.66a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.66a
// CHECK:STDOUT:   %T.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.4b6
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.e1c
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.179
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.059
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.66a) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.66a
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.c6b
// CHECK:STDOUT:   %Inner => constants.%Inner.4b6
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.6f1
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.f69
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.059
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.179
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.afe
// CHECK:STDOUT:   %.loc9_38 => constants.%.a5f
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.32d
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.c4a
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.696: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.455: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.44f: %Inner.type.455 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.371: type = facet_access_type %Self.44f [symbolic]
// CHECK:STDOUT:   %pattern_type.eb9: type = pattern_type %Self.as_type.371 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.184: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Self.44f) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.e16: %Inner.WithSelf.F.type.184 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.cde: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8b1: %Inner.assoc_type.cde = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %C.2eb: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.610: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.607: <witness> = require_complete_type %Inner.type.455 [symbolic]
// CHECK:STDOUT:   %pattern_type.ec2: type = pattern_type %C.2eb [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.fe6: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.ced: %C.as.Inner.impl.F.type.fe6 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.79f: %Inner.type.455 = facet_value %C.2eb, (%Inner.impl_witness.610) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.5aa: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.79f) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.c96: %Inner.WithSelf.F.type.5aa = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.7c6: <witness> = require_complete_type %C.2eb [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.2eb, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %.f29: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.769: %Inner.type.455 = facet_value %C.2eb, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.5c6: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.769) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.505: %Inner.WithSelf.F.type.5c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %.395: type = fn_type_with_self_type %Inner.WithSelf.F.type.5c6, %Inner.facet.769 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.395 = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Inner.WithSelf.F(%T, %Inner.facet.769) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.c71: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.dfb: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.8e0: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.690: <witness> = impl_witness @D.as.Inner.impl.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.769: %Inner.type.dfb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.8aa: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Self.44f) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.fc0: %Inner.WithSelf.F.type.8aa = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.02a: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.0d5: %Inner.assoc_type.02a = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.311: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F.type: type = fn_type @D.as.Inner.impl.F [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F: %D.as.Inner.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.f6d: %Inner.type.dfb = facet_value %D, (%Inner.impl_witness.690) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.3e7: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.f6d) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.a6f: %Inner.WithSelf.F.type.3e7 = struct_value () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.66f: type = pattern_type %C.8e0 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C.8e0 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.eed: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.ee0: <witness> = complete_type_witness %Inner.type.dfb [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.3df: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.c32: %C.as.Inner.impl.F.type.3df = struct_value () [concrete]
// CHECK:STDOUT:   %.d1e: require_specific_def_type = require_specific_def @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %Inner.facet.d0e: %Inner.type.dfb = facet_value %C.8e0, (%Inner.impl_witness.eed) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.e49: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.d0e) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.383: %Inner.WithSelf.F.type.e49 = struct_value () [concrete]
// CHECK:STDOUT:   %.9f6: type = fn_type_with_self_type %Inner.WithSelf.F.type.e49, %Inner.facet.d0e [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.c32, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc23_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc22 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22: Core.Form = init_form %i32.loc22 [concrete = constants.%.70c]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:   %Self.loc5_19.2: @Inner.%Inner.type (%Inner.type.455) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.44f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.455) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.44f)]
// CHECK:STDOUT:     %Inner.WithSelf.decl = interface_with_self_decl @Inner [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Inner.WithSelf.F.decl: @Inner.WithSelf.%Inner.WithSelf.F.type (%Inner.WithSelf.F.type.184) = fn_decl @Inner.WithSelf.F [symbolic = @Inner.WithSelf.%Inner.WithSelf.F (constants.%Inner.WithSelf.F.e16)] {
// CHECK:STDOUT:       %self.param_patt: @Inner.WithSelf.F.%pattern_type.loc6_14 (%pattern_type.eb9) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Inner.WithSelf.F.%pattern_type.loc6_14 (%pattern_type.eb9) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.WithSelf.F.%pattern_type.loc6_27 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.WithSelf.F.%pattern_type.loc6_27 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_27.2: Core.Form = init_form %T.ref [symbolic = %.loc6_27.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.371) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %.loc6_16.3 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.371)] {
// CHECK:STDOUT:         %.loc6_16.2: @Inner.WithSelf.F.%Inner.type (%Inner.type.455) = specific_constant @Inner.%Self.loc5_19.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.44f)]
// CHECK:STDOUT:         %Self.ref: @Inner.WithSelf.F.%Inner.type (%Inner.type.455) = name_ref Self, %.loc6_16.2 [symbolic = %Self (constants.%Self.44f)]
// CHECK:STDOUT:         %Self.as_type.loc6_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.371)]
// CHECK:STDOUT:         %.loc6_16.3: type = converted %Self.ref, %Self.as_type.loc6_16.2 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.371)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.371) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Inner.WithSelf.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Inner.WithSelf.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_28.1: @Inner.WithSelf.%Inner.assoc_type (%Inner.assoc_type.cde) = assoc_entity element0, %Inner.WithSelf.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.8b1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_19.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @Inner.WithSelf.%assoc0.loc6_28.1
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .Inner = <poisoned>
// CHECK:STDOUT:     .D = <poisoned>
// CHECK:STDOUT:     witness = (@Inner.WithSelf.%Inner.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Inner.impl(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.2eb)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.610)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.607)]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.fe6)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.fe6) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.ced)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.fe6) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.ced)] {
// CHECK:STDOUT:       %self.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.ec2) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.ec2) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_26 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @C.as.Inner.impl.F.%pattern_type.loc11_26 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc11_26.3: Core.Form = init_form %T.ref [symbolic = %.loc11_26.2 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @C.as.Inner.impl.F.%C (%C.2eb) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.2eb)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T) [symbolic = %C (constants.%C.2eb)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.2eb)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Inner.impl.F.%C (%C.2eb) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @C.as.Inner.impl.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @C.as.Inner.impl.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness.loc10_19.1: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.610)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %C.as.Inner.impl.F.decl
// CHECK:STDOUT:     .Inner = <poisoned>
// CHECK:STDOUT:     extend %Inner.ref
// CHECK:STDOUT:     witness = %Inner.impl_witness.loc10_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Inner.impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:   %D.as.Inner.impl.F.decl: %D.as.Inner.impl.F.type = fn_decl @D.as.Inner.impl.F [concrete = constants.%D.as.Inner.impl.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.311 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.311 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (%D.as.Inner.impl.F.decl), @D.as.Inner.impl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.690]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .F = %D.as.Inner.impl.F.decl
// CHECK:STDOUT:   extend %Inner.ref
// CHECK:STDOUT:   witness = %Inner.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_14.1)> [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_14.1) [symbolic = %C (constants.%C.2eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.455)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.2eb)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.696
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @C.as.Inner.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.2eb [symbolic = %C (constants.%C.2eb)]
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2eb
// CHECK:STDOUT:     .Inner = <poisoned>
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   impl_decl @D.as.Inner.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.c71]
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.dfb]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.dfb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.WithSelf.F(@Outer.%T.loc4_14.2: type, @Inner.%Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.455)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:   %Self: @Inner.WithSelf.F.%Inner.type (%Inner.type.455) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.44f)]
// CHECK:STDOUT:   %Self.as_type.loc6_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.371)]
// CHECK:STDOUT:   %pattern_type.loc6_14: type = pattern_type %Self.as_type.loc6_16.1 [symbolic = %pattern_type.loc6_14 (constants.%pattern_type.eb9)]
// CHECK:STDOUT:   %.loc6_27.1: Core.Form = init_form %T [symbolic = %.loc6_27.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_27: type = pattern_type %T [symbolic = %pattern_type.loc6_27 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.371)) -> out %return.param: @Inner.WithSelf.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.2eb)]
// CHECK:STDOUT:   %pattern_type.loc11_16: type = pattern_type %C [symbolic = %pattern_type.loc11_16 (constants.%pattern_type.ec2)]
// CHECK:STDOUT:   %.loc11_26.2: Core.Form = init_form %T [symbolic = %.loc11_26.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc11_26: type = pattern_type %T [symbolic = %pattern_type.loc11_26 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.7c6)]
// CHECK:STDOUT:   %require_complete.loc11_23: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_23 (constants.%require_complete.944)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:   %require_complete.loc11_48: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_48 (constants.%require_complete.607)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.cde)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.cde) = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.8b1)]
// CHECK:STDOUT:   %.loc11_41.1: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic = %.loc11_41.1 (constants.%.f29)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.455) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.769)]
// CHECK:STDOUT:   %Inner.WithSelf.F.type: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet) [symbolic = %Inner.WithSelf.F.type (constants.%Inner.WithSelf.F.type.5c6)]
// CHECK:STDOUT:   %.loc11_41.2: type = fn_type_with_self_type %Inner.WithSelf.F.type, %Inner.facet [symbolic = %.loc11_41.2 (constants.%.395)]
// CHECK:STDOUT:   %impl.elem0.loc11_41.2: @C.as.Inner.impl.F.%.loc11_41.2 (%.395) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2: <specific function> = specific_impl_function %impl.elem0.loc11_41.2, @Inner.WithSelf.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.2eb)) -> out %return.param: @C.as.Inner.impl.F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.2eb) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_43: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_43 [symbolic = %Inner.type (constants.%Inner.type.455)]
// CHECK:STDOUT:     %.loc11_48: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.cde) = specific_constant @Inner.WithSelf.%assoc0.loc6_28.1, @Inner.WithSelf(constants.%T, constants.%Self.44f) [symbolic = %assoc0 (constants.%assoc0.8b1)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.cde) = name_ref F, %.loc11_48 [symbolic = %assoc0 (constants.%assoc0.8b1)]
// CHECK:STDOUT:     %impl.elem0.loc11_41.1: @C.as.Inner.impl.F.%.loc11_41.2 (%.395) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc11_41: <bound method> = bound_method %self.ref, %impl.elem0.loc11_41.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_41.1: <specific function> = specific_impl_function %impl.elem0.loc11_41.1, @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.769) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc11_52: <bound method> = bound_method %self.ref, %specific_impl_fn.loc11_41.1
// CHECK:STDOUT:     %.loc11_26.1: ref @C.as.Inner.impl.F.%T (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %Inner.WithSelf.F.call: init @C.as.Inner.impl.F.%T (%T) to %.loc11_26.1 = call %bound_method.loc11_52(%self.ref)
// CHECK:STDOUT:     return %Inner.WithSelf.F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Inner.impl.F(%self.param: %D) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.66f = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.66f = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.8e0 = var %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_26.2: init %C.8e0 to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_3: init %C.8e0 = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_3
// CHECK:STDOUT:   %.loc23_20.1: type = splice_block %C.ref [concrete = constants.%C.8e0] {
// CHECK:STDOUT:     %Outer.ref.loc23: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc23: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.c71]
// CHECK:STDOUT:     %.loc23_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.8e0]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc23_20.2 [concrete = constants.%C.8e0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.8e0 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.8e0 = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.c71]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.dfb]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.dfb]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.02a = specific_constant @Inner.WithSelf.%assoc0.loc6_28.1, @Inner.WithSelf(constants.%i32, constants.%Self.44f) [concrete = constants.%assoc0.0d5]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.02a = name_ref F, %.loc24_29 [concrete = constants.%assoc0.0d5]
// CHECK:STDOUT:   %impl.elem0: %.9f6 = impl_witness_access constants.%Inner.impl_witness.eed, element0 [concrete = constants.%C.as.Inner.impl.F.c32]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.8e0 = acquire_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%c.var)
// CHECK:STDOUT:   return %C.as.Inner.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.2(%self.param: ref %C.8e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %C => constants.%C.2eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.44f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Self.44f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Self => constants.%Self.44f
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.184
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.e16
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.cde
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.8b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Self.44f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Self => constants.%Self.44f
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%Self.as_type.371
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.eb9
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.2eb
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.610
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.607
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.fe6
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.ced
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.2eb
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.ec2
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.79f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Self => constants.%Inner.facet.79f
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.5aa
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.c96
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.cde
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.8b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.79f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Self => constants.%Inner.facet.79f
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.2eb
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.ec2
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.769) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Self => constants.%Inner.facet.769
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.5c6
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.505
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.cde
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.8b1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.769) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.455
// CHECK:STDOUT:   %Self => constants.%Inner.facet.769
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.2eb
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.ec2
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %C => constants.%C.8e0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.769
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Self.44f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %Self => constants.%Self.44f
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.8aa
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.fc0
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.02a
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.0d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.f6d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %Self => constants.%Inner.facet.f6d
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.3e7
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.a6f
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.02a
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.0d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.f6d) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %Self => constants.%Inner.facet.f6d
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.311
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.70c
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.c6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.8e0
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.eed
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.ee0
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.3df
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.c32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.d0e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %Self => constants.%Inner.facet.d0e
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.e49
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.383
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.02a
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.0d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.8e0
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.66f
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.70c
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.c6b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_23 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%complete_type.ee0
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.02a
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.0d5
// CHECK:STDOUT:   %.loc11_41.1 => constants.%.d1e
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.eed
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.d0e
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.e49
// CHECK:STDOUT:   %.loc11_41.2 => constants.%.9f6
// CHECK:STDOUT:   %impl.elem0.loc11_41.2 => constants.%C.as.Inner.impl.F.c32
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.d0e) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.dfb
// CHECK:STDOUT:   %Self => constants.%Inner.facet.d0e
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.8e0
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.66f
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.70c
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.c6b
// CHECK:STDOUT: }
// CHECK:STDOUT:
