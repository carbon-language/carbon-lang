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
// CHECK:STDOUT:   %pattern_type.b72: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.467: type = class_type @Outer, @Outer(%T.f49) [symbolic]
// CHECK:STDOUT:   %Inner.c8f: type = class_type @Inner, @Inner(%T.f49) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f49 [symbolic]
// CHECK:STDOUT:   %require_complete.77e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.c0fdca.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.elem.36d: type = unbound_element_type %Inner.c8f, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n.fa5: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.c3a: <witness> = complete_type_witness %struct_type.n.fa5 [symbolic]
// CHECK:STDOUT:   %.6d6: Core.Form = init_form %Inner.c8f [symbolic]
// CHECK:STDOUT:   %pattern_type.f8c: type = pattern_type %Inner.c8f [symbolic]
// CHECK:STDOUT:   %Outer.F.type.4f4: type = fn_type @Outer.F, @Outer(%T.f49) [symbolic]
// CHECK:STDOUT:   %Outer.F.51f: %Outer.F.type.4f4 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.93b: <witness> = require_complete_type %Inner.c8f [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2dd: <witness> = lookup_impl_witness %T.f49, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e9f10f.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f49) [symbolic]
// CHECK:STDOUT:   %.e16: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e9f10f.2, %T.f49 [symbolic]
// CHECK:STDOUT:   %impl.elem0.41e: %.e16 = impl_witness_access %Copy.lookup_impl_witness.2dd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7ec: <specific function> = specific_impl_function %impl.elem0.41e, @Copy.WithSelf.Op(%T.f49) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.14d: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Outer.0c2: type = class_type @Outer, @Outer(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %Inner.cc9: type = class_type @Inner, @Inner(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %Outer.F.type.3a5: type = fn_type @Outer.F, @Outer(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %Outer.F.a82: %Outer.F.type.3a5 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.8cd: type = unbound_element_type %Inner.cc9, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.c7a: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.edf: <witness> = complete_type_witness %struct_type.n.c7a [concrete]
// CHECK:STDOUT:   %pattern_type.37d: type = pattern_type %Inner.cc9 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.061: <witness> = impl_witness imports.%Copy.impl_witness_table.fb5 [concrete]
// CHECK:STDOUT:   %Copy.facet.23a: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.061) [concrete]
// CHECK:STDOUT:   %.e0c: Core.Form = init_form %Inner.cc9 [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.a82, @Outer.F(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d9c: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet.14d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.4: type = fn_type @Destroy.Op.loc13_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.4: %Destroy.Op.type.790b3b.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.83f = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.fb5 = impl_witness_table (%Core.import_ref.83f), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.b72 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.f49)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32.loc12 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32.loc12 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc4_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.f49)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_14.1) [symbolic = %Inner (constants.%Inner.c8f)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_14.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.4f4)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.4f4) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.51f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.c8f)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.4f4) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.51f)] {
// CHECK:STDOUT:       %n.param_patt: @Outer.F.%pattern_type.loc9_9 (%pattern_type.c0fdca.2) = value_param_pattern [concrete]
// CHECK:STDOUT:       %n.patt: @Outer.F.%pattern_type.loc9_9 (%pattern_type.c0fdca.2) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Outer.F.%pattern_type.loc9_17 (%pattern_type.f8c) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Outer.F.%pattern_type.loc9_17 (%pattern_type.f8c) = return_slot_pattern %return.param_patt, %Inner.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.f49) [symbolic = %Inner (constants.%Inner.c8f)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17.2 [symbolic = %Inner (constants.%Inner.c8f)]
// CHECK:STDOUT:       %.loc9_17.3: Core.Form = init_form %Inner.ref [symbolic = %.loc9_17.1 (constants.%.6d6)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:         %T.as_type.loc9_11.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc9_11.2: type = converted %T.ref, %T.as_type.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.c8f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.c8f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.467
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.77e)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.c8f)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.as_type [symbolic = %Inner.elem (constants.%Inner.elem.36d)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.as_type (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.fa5)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.c3a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6: @Inner.%Inner.elem (%Inner.elem.36d) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.fa5 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.c3a)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.c8f
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:   %T.as_type.loc9_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_9: type = pattern_type %T.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_9 (constants.%pattern_type.c0fdca.2)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.c8f)]
// CHECK:STDOUT:   %.loc9_17.1: Core.Form = init_form %Inner [symbolic = %.loc9_17.1 (constants.%.6d6)]
// CHECK:STDOUT:   %pattern_type.loc9_17: type = pattern_type %Inner [symbolic = %pattern_type.loc9_17 (constants.%pattern_type.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.as_type.loc9_11.1 [symbolic = %require_complete.loc9_9 (constants.%require_complete.77e)]
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.93b)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.fa5)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.e9f10f.2)]
// CHECK:STDOUT:   %.loc9_38: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_38 (constants.%.e16)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2dd)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38 (%.e16) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)) -> out %return.param: @Outer.F.%Inner (%Inner.c8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.fa5) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38 (%.e16) = impl_witness_access constants.%Copy.lookup_impl_witness.2dd, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.WithSelf.Op(constants.%T.f49) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = class_element_access %return.param, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = call %bound_method.loc9_38.2(%n.ref)
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.c8f) to %return.param = class_init (%.loc9_39.3)
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.c8f) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.37d = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.37d = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.cc9 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc13_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value %i32.loc13_35, (constants.%Copy.impl_witness.1a4) [concrete = constants.%Copy.facet.14d]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.14d]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.14d) [concrete = constants.%Outer.0c2]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.3a5 = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.14d) [concrete = constants.%Outer.F.a82]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.3a5 = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.a82]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.061) [concrete = constants.%Copy.facet.23a]
// CHECK:STDOUT:   %.loc13_43: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43 [concrete = constants.%Copy.facet.23a]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.14d) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3: ref %Inner.cc9 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0.loc13: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.d9c]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.cc9 to %.loc13_3 = call %Outer.F.specific_fn(%.loc13_42.2)
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.cc9] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc13_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value %i32.loc13_16, (constants.%Copy.impl_witness.1a4) [concrete = constants.%Copy.facet.14d]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.14d]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.14d) [concrete = constants.%Outer.0c2]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.14d) [concrete = constants.%Inner.cc9]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.cc9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.cc9 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.cc9 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.8cd = name_ref n, @Inner.%.loc6 [concrete = @Inner.%.loc6]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = acquire_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0.loc14: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn.loc14
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.2dd605.4
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
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.3(%self.param: ref %struct_type.n.c7a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.4(%self.param: ref %Inner.cc9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.f49) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.f49
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.c8f
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.4f4
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.51f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.f49) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.77e
// CHECK:STDOUT:   %Inner => constants.%Inner.c8f
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.36d
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.fa5
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.c3a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.f49) {
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.c0fdca.2
// CHECK:STDOUT:   %Inner => constants.%Inner.c8f
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.6d6
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.f8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.14d) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.14d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.cc9
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.3a5
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.a82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.14d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.14d
// CHECK:STDOUT:   %T.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.cc9
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.8cd
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.c7a
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.edf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.14d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.14d
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.104
// CHECK:STDOUT:   %Inner => constants.%Inner.cc9
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.e0c
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.37d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.edf
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.c7a
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc9_38 => constants.%.604
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.1a4
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.ce7
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
// CHECK:STDOUT:   %Outer.3b5: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.4f1: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.381: %Inner.type.4f1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.115: type = facet_access_type %Self.381 [symbolic]
// CHECK:STDOUT:   %pattern_type.87e: type = pattern_type %Self.as_type.115 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.f3a: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Self.381) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.d63: %Inner.WithSelf.F.type.f3a = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.126: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.9af: %Inner.assoc_type.126 = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %C.514: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.3ba: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.41e: <witness> = require_complete_type %Inner.type.4f1 [symbolic]
// CHECK:STDOUT:   %pattern_type.55d: type = pattern_type %C.514 [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.5c2: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.8eb: %C.as.Inner.impl.F.type.5c2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.b3f: %Inner.type.4f1 = facet_value %C.514, (%Inner.impl_witness.3ba) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.922: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.b3f) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.f36: %Inner.WithSelf.F.type.922 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.d21: <witness> = require_complete_type %C.514 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.514, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %.f4b: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.629: %Inner.type.4f1 = facet_value %C.514, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.550: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.629) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.b8b: %Inner.WithSelf.F.type.550 = struct_value () [symbolic]
// CHECK:STDOUT:   %.cb2: type = fn_type_with_self_type %Inner.WithSelf.F.type.550, %Inner.facet.629 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.cb2 = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Inner.WithSelf.F(%T, %Inner.facet.629) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.f42: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.d1b: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.525: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.38e: <witness> = impl_witness @D.as.Inner.impl.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.693: %Inner.type.d1b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.cf7: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Self.381) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.9bb: %Inner.WithSelf.F.type.cf7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.7bd: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.c3f: %Inner.assoc_type.7bd = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.f1a: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F.type: type = fn_type @D.as.Inner.impl.F [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F: %D.as.Inner.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.072: %Inner.type.d1b = facet_value %D, (%Inner.impl_witness.38e) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.6d5: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.072) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.2ef: %Inner.WithSelf.F.type.6d5 = struct_value () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.abe: type = pattern_type %C.525 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C.525 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.4a5: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.15f: <witness> = complete_type_witness %Inner.type.d1b [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.8de: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.40e: %C.as.Inner.impl.F.type.8de = struct_value () [concrete]
// CHECK:STDOUT:   %.b2e: require_specific_def_type = require_specific_def @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %Inner.facet.51d: %Inner.type.d1b = facet_value %C.525, (%Inner.impl_witness.4a5) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.8fb: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.51d) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.723: %Inner.WithSelf.F.type.8fb = struct_value () [concrete]
// CHECK:STDOUT:   %.633: type = fn_type_with_self_type %Inner.WithSelf.F.type.8fb, %Inner.facet.51d [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.40e, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc23_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32.loc22 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22: Core.Form = init_form %i32.loc22 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:   %Self.loc5_19.2: @Inner.%Inner.type (%Inner.type.4f1) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.381)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.4f1) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.381)]
// CHECK:STDOUT:     %Inner.WithSelf.decl = interface_with_self_decl @Inner [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Inner.WithSelf.F.decl: @Inner.WithSelf.%Inner.WithSelf.F.type (%Inner.WithSelf.F.type.f3a) = fn_decl @Inner.WithSelf.F [symbolic = @Inner.WithSelf.%Inner.WithSelf.F (constants.%Inner.WithSelf.F.d63)] {
// CHECK:STDOUT:       %self.param_patt: @Inner.WithSelf.F.%pattern_type.loc6_14 (%pattern_type.87e) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Inner.WithSelf.F.%pattern_type.loc6_14 (%pattern_type.87e) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.WithSelf.F.%pattern_type.loc6_27 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.WithSelf.F.%pattern_type.loc6_27 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_27.2: Core.Form = init_form %T.ref [symbolic = %.loc6_27.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.115) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %.loc6_16.3 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.115)] {
// CHECK:STDOUT:         %.loc6_16.2: @Inner.WithSelf.F.%Inner.type (%Inner.type.4f1) = specific_constant @Inner.%Self.loc5_19.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.381)]
// CHECK:STDOUT:         %Self.ref: @Inner.WithSelf.F.%Inner.type (%Inner.type.4f1) = name_ref Self, %.loc6_16.2 [symbolic = %Self (constants.%Self.381)]
// CHECK:STDOUT:         %Self.as_type.loc6_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.115)]
// CHECK:STDOUT:         %.loc6_16.3: type = converted %Self.ref, %Self.as_type.loc6_16.2 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.115)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.115) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Inner.WithSelf.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Inner.WithSelf.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_28.1: @Inner.WithSelf.%Inner.assoc_type (%Inner.assoc_type.126) = assoc_entity element0, %Inner.WithSelf.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.9af)]
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.514)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.3ba)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.41e)]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.5c2)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.5c2) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.8eb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.5c2) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.8eb)] {
// CHECK:STDOUT:       %self.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.55d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.55d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_26 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @C.as.Inner.impl.F.%pattern_type.loc11_26 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc11_26.3: Core.Form = init_form %T.ref [symbolic = %.loc11_26.2 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @C.as.Inner.impl.F.%C (%C.514) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.514)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T) [symbolic = %C (constants.%C.514)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.514)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Inner.impl.F.%C (%C.514) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @C.as.Inner.impl.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @C.as.Inner.impl.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness.loc10_19.1: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.3ba)]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f1a = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.f1a = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (%D.as.Inner.impl.F.decl), @D.as.Inner.impl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.38e]
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
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_14.1)> [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_14.1) [symbolic = %C (constants.%C.514)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.4f1)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.514)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.3b5
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
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.514 [symbolic = %C (constants.%C.514)]
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.514
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
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.f42]
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.d1b]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.d1b]
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
// CHECK:STDOUT: generic fn @Inner.WithSelf.F(@Outer.%T.loc4_14.2: type, @Inner.%Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.4f1)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:   %Self: @Inner.WithSelf.F.%Inner.type (%Inner.type.4f1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.381)]
// CHECK:STDOUT:   %Self.as_type.loc6_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.115)]
// CHECK:STDOUT:   %pattern_type.loc6_14: type = pattern_type %Self.as_type.loc6_16.1 [symbolic = %pattern_type.loc6_14 (constants.%pattern_type.87e)]
// CHECK:STDOUT:   %.loc6_27.1: Core.Form = init_form %T [symbolic = %.loc6_27.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_27: type = pattern_type %T [symbolic = %pattern_type.loc6_27 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.115)) -> out %return.param: @Inner.WithSelf.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.514)]
// CHECK:STDOUT:   %pattern_type.loc11_16: type = pattern_type %C [symbolic = %pattern_type.loc11_16 (constants.%pattern_type.55d)]
// CHECK:STDOUT:   %.loc11_26.2: Core.Form = init_form %T [symbolic = %.loc11_26.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc11_26: type = pattern_type %T [symbolic = %pattern_type.loc11_26 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.d21)]
// CHECK:STDOUT:   %require_complete.loc11_23: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_23 (constants.%require_complete.944)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:   %require_complete.loc11_48: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_48 (constants.%require_complete.41e)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.126)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.126) = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.9af)]
// CHECK:STDOUT:   %.loc11_41.1: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic = %.loc11_41.1 (constants.%.f4b)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.4f1) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.629)]
// CHECK:STDOUT:   %Inner.WithSelf.F.type: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet) [symbolic = %Inner.WithSelf.F.type (constants.%Inner.WithSelf.F.type.550)]
// CHECK:STDOUT:   %.loc11_41.2: type = fn_type_with_self_type %Inner.WithSelf.F.type, %Inner.facet [symbolic = %.loc11_41.2 (constants.%.cb2)]
// CHECK:STDOUT:   %impl.elem0.loc11_41.2: @C.as.Inner.impl.F.%.loc11_41.2 (%.cb2) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2: <specific function> = specific_impl_function %impl.elem0.loc11_41.2, @Inner.WithSelf.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.514)) -> out %return.param: @C.as.Inner.impl.F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.514) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_43: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_43 [symbolic = %Inner.type (constants.%Inner.type.4f1)]
// CHECK:STDOUT:     %.loc11_48: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.126) = specific_constant @Inner.WithSelf.%assoc0.loc6_28.1, @Inner.WithSelf(constants.%T, constants.%Self.381) [symbolic = %assoc0 (constants.%assoc0.9af)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.126) = name_ref F, %.loc11_48 [symbolic = %assoc0 (constants.%assoc0.9af)]
// CHECK:STDOUT:     %impl.elem0.loc11_41.1: @C.as.Inner.impl.F.%.loc11_41.2 (%.cb2) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc11_41: <bound method> = bound_method %self.ref, %impl.elem0.loc11_41.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_41.1: <specific function> = specific_impl_function %impl.elem0.loc11_41.1, @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.629) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.abe = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.abe = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.525 = var %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_26.2: init %C.525 to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_3: init %C.525 = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_3
// CHECK:STDOUT:   %.loc23_20.1: type = splice_block %C.ref [concrete = constants.%C.525] {
// CHECK:STDOUT:     %Outer.ref.loc23: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc23: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.f42]
// CHECK:STDOUT:     %.loc23_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.525]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc23_20.2 [concrete = constants.%C.525]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.525 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.525 = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.f42]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.d1b]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.d1b]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.7bd = specific_constant @Inner.WithSelf.%assoc0.loc6_28.1, @Inner.WithSelf(constants.%i32, constants.%Self.381) [concrete = constants.%assoc0.c3f]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.7bd = name_ref F, %.loc24_29 [concrete = constants.%assoc0.c3f]
// CHECK:STDOUT:   %impl.elem0: %.633 = impl_witness_access constants.%Inner.impl_witness.4a5, element0 [concrete = constants.%C.as.Inner.impl.F.40e]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.525 = acquire_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.2dd605.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%c.var)
// CHECK:STDOUT:   return %C.as.Inner.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.2(%self.param: ref %C.525) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %C => constants.%C.514
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.381
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Self.381) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Self => constants.%Self.381
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.f3a
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.d63
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.126
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.9af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Self.381) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Self => constants.%Self.381
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%Self.as_type.115
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.87e
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
// CHECK:STDOUT:   %C => constants.%C.514
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.3ba
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.41e
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.5c2
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.8eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.514
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.55d
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.b3f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Self => constants.%Inner.facet.b3f
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.922
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.f36
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.126
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.9af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.b3f) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Self => constants.%Inner.facet.b3f
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.514
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.55d
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.629) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Self => constants.%Inner.facet.629
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.550
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.b8b
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.126
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.9af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.629) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.4f1
// CHECK:STDOUT:   %Self => constants.%Inner.facet.629
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.514
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.55d
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %C => constants.%C.525
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.693
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Self.381) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %Self => constants.%Self.381
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.cf7
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.9bb
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.7bd
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.c3f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.072) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.072
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.6d5
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.2ef
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.7bd
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.c3f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.072) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.072
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.f1a
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.7ea
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.525
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.4a5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.15f
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.8de
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.40e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.51d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.51d
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.8fb
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.723
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.7bd
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.c3f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.525
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.abe
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.7ea
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.104
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_23 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%complete_type.15f
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.7bd
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c3f
// CHECK:STDOUT:   %.loc11_41.1 => constants.%.b2e
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.4a5
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.51d
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.8fb
// CHECK:STDOUT:   %.loc11_41.2 => constants.%.633
// CHECK:STDOUT:   %impl.elem0.loc11_41.2 => constants.%C.as.Inner.impl.F.40e
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.51d) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d1b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.51d
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.525
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.abe
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.7ea
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.104
// CHECK:STDOUT: }
// CHECK:STDOUT:
