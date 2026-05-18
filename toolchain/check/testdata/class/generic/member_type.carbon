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
// CHECK:STDOUT:   %pattern_type.6c0: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.0e9: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.418: type = class_type @Outer, @Outer(%T.0e9) [symbolic]
// CHECK:STDOUT:   %Inner.9f6: type = class_type @Inner, @Inner(%T.0e9) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.0e9 [symbolic]
// CHECK:STDOUT:   %require_complete.8de: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.ecae91.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.elem.542: type = unbound_element_type %Inner.9f6, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n.b4a: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.1d7: <witness> = complete_type_witness %struct_type.n.b4a [symbolic]
// CHECK:STDOUT:   %.9a1: Core.Form = init_form %Inner.9f6 [symbolic]
// CHECK:STDOUT:   %pattern_type.4e6: type = pattern_type %Inner.9f6 [symbolic]
// CHECK:STDOUT:   %Outer.F.type.70a: type = fn_type @Outer.F, @Outer(%T.0e9) [symbolic]
// CHECK:STDOUT:   %Outer.F.fd0: %Outer.F.type.70a = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.c9d: <witness> = require_complete_type %Inner.9f6 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cf0: <witness> = lookup_impl_witness %T.0e9, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e87e4c.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.0e9) [symbolic]
// CHECK:STDOUT:   %.499: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e87e4c.2, %T.0e9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.615: %.499 = impl_witness_access %Copy.lookup_impl_witness.cf0, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.baf: <specific function> = specific_impl_function %impl.elem0.615, @Copy.WithSelf.Op(%T.0e9) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.f87: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.521: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.783: %Int.as.Copy.impl.Op.type.521 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f1f: <witness> = impl_witness imports.%Copy.impl_witness_table.592, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f98: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6d3: %Int.as.Copy.impl.Op.type.f98 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.ddd: %Copy.type = facet_value %i32, (%Copy.impl_witness.f1f) [concrete]
// CHECK:STDOUT:   %Outer.b09: type = class_type @Outer, @Outer(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %Inner.bd1: type = class_type @Inner, @Inner(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %Outer.F.type.e4e: type = fn_type @Outer.F, @Outer(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %Outer.F.915: %Outer.F.type.e4e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.b0a: type = unbound_element_type %Inner.bd1, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.88c: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.14a: <witness> = complete_type_witness %struct_type.n.88c [concrete]
// CHECK:STDOUT:   %pattern_type.56b: type = pattern_type %Inner.bd1 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.55e: <witness> = impl_witness imports.%Copy.impl_witness_table.e48 [concrete]
// CHECK:STDOUT:   %Copy.facet.f9e: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.55e) [concrete]
// CHECK:STDOUT:   %.da1: Core.Form = init_form %Inner.bd1 [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.915, @Outer.F(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ef: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.abb: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f4c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e6c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ca = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.0ef = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f4c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.61c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.cbf: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.61c, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.d72: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.83a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.ddd) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %Copy.WithSelf.Op.type.83a, %Copy.facet.ddd [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.6d3, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.4: type = fn_type @Destroy.Op.loc13_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.4: %Destroy.Op.type.3f5f78.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.0bb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.521) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.783)]
// CHECK:STDOUT:   %Copy.impl_witness_table.592 = impl_witness_table (%Core.import_ref.0bb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f66 = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.e48 = impl_witness_table (%Core.import_ref.f66), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.fa9: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.aad) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.abb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e6c = impl_witness_table (%Core.import_ref.fa9), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.6c0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.0e9)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.17a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.17a = return_slot_pattern %return.param_patt, %i32.loc12 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32.loc12 [concrete = constants.%.f87]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc4_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.0e9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_14.1) [symbolic = %Inner (constants.%Inner.9f6)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_14.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.70a)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.70a) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.fd0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.9f6)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.70a) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.fd0)] {
// CHECK:STDOUT:       %n.param_patt: @Outer.F.%pattern_type.loc9_9 (%pattern_type.ecae91.2) = value_param_pattern [concrete]
// CHECK:STDOUT:       %n.patt: @Outer.F.%pattern_type.loc9_9 (%pattern_type.ecae91.2) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Outer.F.%pattern_type.loc9_17 (%pattern_type.4e6) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Outer.F.%pattern_type.loc9_17 (%pattern_type.4e6) = return_slot_pattern %return.param_patt, %Inner.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.0e9) [symbolic = %Inner (constants.%Inner.9f6)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17.2 [symbolic = %Inner (constants.%Inner.9f6)]
// CHECK:STDOUT:       %.loc9_17.3: Core.Form = init_form %Inner.ref [symbolic = %.loc9_17.1 (constants.%.9a1)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T.0e9)]
// CHECK:STDOUT:         %T.as_type.loc9_11.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc9_11.2: type = converted %T.ref, %T.as_type.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.9f6) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.9f6) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.418
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.0e9)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.8de)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.9f6)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.as_type [symbolic = %Inner.elem (constants.%Inner.elem.542)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.as_type (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.b4a)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.1d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6: @Inner.%Inner.elem (%Inner.elem.542) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.b4a [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.1d7)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.9f6
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.0e9)]
// CHECK:STDOUT:   %T.as_type.loc9_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_9: type = pattern_type %T.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_9 (constants.%pattern_type.ecae91.2)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.9f6)]
// CHECK:STDOUT:   %.loc9_17.1: Core.Form = init_form %Inner [symbolic = %.loc9_17.1 (constants.%.9a1)]
// CHECK:STDOUT:   %pattern_type.loc9_17: type = pattern_type %Inner [symbolic = %pattern_type.loc9_17 (constants.%pattern_type.4e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.as_type.loc9_11.1 [symbolic = %require_complete.loc9_9 (constants.%require_complete.8de)]
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.c9d)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.b4a)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.e87e4c.2)]
// CHECK:STDOUT:   %.loc9_38: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_38 (constants.%.499)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cf0)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38 (%.499) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.615)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.baf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)) -> out %return.param: @Outer.F.%Inner (%Inner.9f6) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.b4a) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38 (%.499) = impl_witness_access constants.%Copy.lookup_impl_witness.cf0, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.615)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.WithSelf.Op(constants.%T.0e9) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.baf)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = class_element_access %return.param, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = call %bound_method.loc9_38.2(%n.ref)
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.9f6) to %return.param = class_init (%.loc9_39.3)
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.9f6) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.56b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.56b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.bd1 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc13_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value %i32.loc13_35, (constants.%Copy.impl_witness.f1f) [concrete = constants.%Copy.facet.ddd]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.ddd]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.ddd) [concrete = constants.%Outer.b09]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.e4e = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.ddd) [concrete = constants.%Outer.F.915]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.e4e = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.915]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.55e) [concrete = constants.%Copy.facet.f9e]
// CHECK:STDOUT:   %.loc13_43: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43 [concrete = constants.%Copy.facet.f9e]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.ddd) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3: ref %Inner.bd1 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0.loc13: %.cbf = impl_witness_access constants.%ImplicitAs.impl_witness.f4c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8a5]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.d72]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.bd1 to %.loc13_3 = call %Outer.F.specific_fn(%.loc13_42.2)
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.bd1] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc13_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value %i32.loc13_16, (constants.%Copy.impl_witness.f1f) [concrete = constants.%Copy.facet.ddd]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.ddd]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.ddd) [concrete = constants.%Outer.b09]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.ddd) [concrete = constants.%Inner.bd1]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.bd1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.bd1 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.bd1 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.b0a = name_ref n, @Inner.%.loc6 [concrete = @Inner.%.loc6]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = acquire_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0.loc14: %.2ec = impl_witness_access constants.%Copy.impl_witness.f1f, element0 [concrete = constants.%Int.as.Copy.impl.Op.6d3]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn.loc14
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.f0ba5f.4
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
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.3(%self.param: ref %struct_type.n.88c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.4(%self.param: ref %Inner.bd1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.0e9) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.0e9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.9f6
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.70a
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.fd0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.0e9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.0e9
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8de
// CHECK:STDOUT:   %Inner => constants.%Inner.9f6
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.542
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.b4a
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.1d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.0e9) {
// CHECK:STDOUT:   %T => constants.%T.0e9
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.ecae91.2
// CHECK:STDOUT:   %Inner => constants.%Inner.9f6
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.9a1
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.4e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.ddd) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.ddd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.bd1
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.e4e
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.915
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.ddd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.ddd
// CHECK:STDOUT:   %T.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.bd1
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.b0a
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.88c
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.14a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.ddd) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.ddd
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.17a
// CHECK:STDOUT:   %Inner => constants.%Inner.bd1
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.da1
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.56b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.14a
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.88c
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.83a
// CHECK:STDOUT:   %.loc9_38 => constants.%.2ec
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f1f
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.6d3
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
// CHECK:STDOUT:   %Outer.6e1: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.549: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.2c3: %Inner.type.549 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.48e: type = facet_access_type %Self.2c3 [symbolic]
// CHECK:STDOUT:   %pattern_type.19c: type = pattern_type %Self.as_type.48e [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.77c: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Self.2c3) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.d05: %Inner.WithSelf.F.type.77c = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.91b: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.ef7: %Inner.assoc_type.91b = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %C.faa: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.539: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.963: <witness> = require_complete_type %Inner.type.549 [symbolic]
// CHECK:STDOUT:   %pattern_type.5c6: type = pattern_type %C.faa [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.8c0: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.451: %C.as.Inner.impl.F.type.8c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.a2e: %Inner.type.549 = facet_value %C.faa, (%Inner.impl_witness.539) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.284: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.a2e) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.493: %Inner.WithSelf.F.type.284 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.fab: <witness> = require_complete_type %C.faa [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.faa, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %.b68: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.b31: %Inner.type.549 = facet_value %C.faa, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.631: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.b31) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.382: %Inner.WithSelf.F.type.631 = struct_value () [symbolic]
// CHECK:STDOUT:   %.2a7: type = fn_type_with_self_type %Inner.WithSelf.F.type.631, %Inner.facet.b31 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.2a7 = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Inner.WithSelf.F(%T, %Inner.facet.b31) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.6bb: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.506: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.ee2: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.d7a: <witness> = impl_witness @D.as.Inner.impl.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.ec7: %Inner.type.506 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.e7b: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Self.2c3) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.545: %Inner.WithSelf.F.type.e7b = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.318: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.fa6: %Inner.assoc_type.318 = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.522: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %.f87: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.17a: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F.type: type = fn_type @D.as.Inner.impl.F [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F: %D.as.Inner.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.5cc: %Inner.type.506 = facet_value %D, (%Inner.impl_witness.d7a) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.e40: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.5cc) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.792: %Inner.WithSelf.F.type.e40 = struct_value () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9e3: type = pattern_type %C.ee2 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C.ee2 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.260: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.228: <witness> = complete_type_witness %Inner.type.506 [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.44c: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.ce3: %C.as.Inner.impl.F.type.44c = struct_value () [concrete]
// CHECK:STDOUT:   %.d3e: require_specific_def_type = require_specific_def @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %Inner.facet.31e: %Inner.type.506 = facet_value %C.ee2, (%Inner.impl_witness.260) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.d8d: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.31e) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.e12: %Inner.WithSelf.F.type.d8d = struct_value () [concrete]
// CHECK:STDOUT:   %.06c: type = fn_type_with_self_type %Inner.WithSelf.F.type.d8d, %Inner.facet.31e [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.ce3, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.3f5f78.2: type = fn_type @Destroy.Op.loc23_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.f0ba5f.2: %Destroy.Op.type.3f5f78.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %return.param_patt: %pattern_type.17a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.17a = return_slot_pattern %return.param_patt, %i32.loc22 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22: Core.Form = init_form %i32.loc22 [concrete = constants.%.f87]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:   %Self.loc5_19.2: @Inner.%Inner.type (%Inner.type.549) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.2c3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.549) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.2c3)]
// CHECK:STDOUT:     %Inner.WithSelf.decl = interface_with_self_decl @Inner [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Inner.WithSelf.F.decl: @Inner.WithSelf.%Inner.WithSelf.F.type (%Inner.WithSelf.F.type.77c) = fn_decl @Inner.WithSelf.F [symbolic = @Inner.WithSelf.%Inner.WithSelf.F (constants.%Inner.WithSelf.F.d05)] {
// CHECK:STDOUT:       %self.param_patt: @Inner.WithSelf.F.%pattern_type.loc6_14 (%pattern_type.19c) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Inner.WithSelf.F.%pattern_type.loc6_14 (%pattern_type.19c) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.WithSelf.F.%pattern_type.loc6_27 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.WithSelf.F.%pattern_type.loc6_27 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_27.2: Core.Form = init_form %T.ref [symbolic = %.loc6_27.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.48e) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %.loc6_16.3 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.48e)] {
// CHECK:STDOUT:         %.loc6_16.2: @Inner.WithSelf.F.%Inner.type (%Inner.type.549) = specific_constant @Inner.%Self.loc5_19.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.2c3)]
// CHECK:STDOUT:         %Self.ref: @Inner.WithSelf.F.%Inner.type (%Inner.type.549) = name_ref Self, %.loc6_16.2 [symbolic = %Self (constants.%Self.2c3)]
// CHECK:STDOUT:         %Self.as_type.loc6_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.48e)]
// CHECK:STDOUT:         %.loc6_16.3: type = converted %Self.ref, %Self.as_type.loc6_16.2 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.48e)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.48e) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Inner.WithSelf.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Inner.WithSelf.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_28.1: @Inner.WithSelf.%Inner.assoc_type (%Inner.assoc_type.91b) = assoc_entity element0, %Inner.WithSelf.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.ef7)]
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.faa)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.539)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.963)]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.8c0)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.8c0) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.451)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.8c0) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.451)] {
// CHECK:STDOUT:       %self.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.5c6) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.5c6) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_26 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @C.as.Inner.impl.F.%pattern_type.loc11_26 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc11_26.3: Core.Form = init_form %T.ref [symbolic = %.loc11_26.2 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @C.as.Inner.impl.F.%C (%C.faa) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.faa)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T) [symbolic = %C (constants.%C.faa)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.faa)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Inner.impl.F.%C (%C.faa) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @C.as.Inner.impl.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @C.as.Inner.impl.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness.loc10_19.1: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.539)]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.522 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.522 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.17a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.17a = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32 [concrete = constants.%.f87]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (%D.as.Inner.impl.F.decl), @D.as.Inner.impl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.d7a]
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
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_14.1)> [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_14.1) [symbolic = %C (constants.%C.faa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.549)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.faa)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.6e1
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
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.faa [symbolic = %C (constants.%C.faa)]
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.faa
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
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.6bb]
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.506]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.506]
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
// CHECK:STDOUT: generic fn @Inner.WithSelf.F(@Outer.%T.loc4_14.2: type, @Inner.%Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.549)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:   %Self: @Inner.WithSelf.F.%Inner.type (%Inner.type.549) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.2c3)]
// CHECK:STDOUT:   %Self.as_type.loc6_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.48e)]
// CHECK:STDOUT:   %pattern_type.loc6_14: type = pattern_type %Self.as_type.loc6_16.1 [symbolic = %pattern_type.loc6_14 (constants.%pattern_type.19c)]
// CHECK:STDOUT:   %.loc6_27.1: Core.Form = init_form %T [symbolic = %.loc6_27.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_27: type = pattern_type %T [symbolic = %pattern_type.loc6_27 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.WithSelf.F.%Self.as_type.loc6_16.1 (%Self.as_type.48e)) -> out %return.param: @Inner.WithSelf.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.faa)]
// CHECK:STDOUT:   %pattern_type.loc11_16: type = pattern_type %C [symbolic = %pattern_type.loc11_16 (constants.%pattern_type.5c6)]
// CHECK:STDOUT:   %.loc11_26.2: Core.Form = init_form %T [symbolic = %.loc11_26.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc11_26: type = pattern_type %T [symbolic = %pattern_type.loc11_26 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.fab)]
// CHECK:STDOUT:   %require_complete.loc11_23: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_23 (constants.%require_complete.944)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:   %require_complete.loc11_48: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_48 (constants.%require_complete.963)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.91b)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.91b) = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.ef7)]
// CHECK:STDOUT:   %.loc11_41.1: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic = %.loc11_41.1 (constants.%.b68)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.549) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.b31)]
// CHECK:STDOUT:   %Inner.WithSelf.F.type: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet) [symbolic = %Inner.WithSelf.F.type (constants.%Inner.WithSelf.F.type.631)]
// CHECK:STDOUT:   %.loc11_41.2: type = fn_type_with_self_type %Inner.WithSelf.F.type, %Inner.facet [symbolic = %.loc11_41.2 (constants.%.2a7)]
// CHECK:STDOUT:   %impl.elem0.loc11_41.2: @C.as.Inner.impl.F.%.loc11_41.2 (%.2a7) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2: <specific function> = specific_impl_function %impl.elem0.loc11_41.2, @Inner.WithSelf.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.faa)) -> out %return.param: @C.as.Inner.impl.F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.faa) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_43: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_43 [symbolic = %Inner.type (constants.%Inner.type.549)]
// CHECK:STDOUT:     %.loc11_48: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.91b) = specific_constant @Inner.WithSelf.%assoc0.loc6_28.1, @Inner.WithSelf(constants.%T, constants.%Self.2c3) [symbolic = %assoc0 (constants.%assoc0.ef7)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.91b) = name_ref F, %.loc11_48 [symbolic = %assoc0 (constants.%assoc0.ef7)]
// CHECK:STDOUT:     %impl.elem0.loc11_41.1: @C.as.Inner.impl.F.%.loc11_41.2 (%.2a7) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc11_41: <bound method> = bound_method %self.ref, %impl.elem0.loc11_41.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_41.1: <specific function> = specific_impl_function %impl.elem0.loc11_41.1, @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.b31) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.9e3 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.9e3 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.ee2 = var %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_26.2: init %C.ee2 to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_3: init %C.ee2 = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_3
// CHECK:STDOUT:   %.loc23_20.1: type = splice_block %C.ref [concrete = constants.%C.ee2] {
// CHECK:STDOUT:     %Outer.ref.loc23: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc23: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.6bb]
// CHECK:STDOUT:     %.loc23_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.ee2]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc23_20.2 [concrete = constants.%C.ee2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.ee2 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.ee2 = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.6bb]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.506]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.506]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.318 = specific_constant @Inner.WithSelf.%assoc0.loc6_28.1, @Inner.WithSelf(constants.%i32, constants.%Self.2c3) [concrete = constants.%assoc0.fa6]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.318 = name_ref F, %.loc24_29 [concrete = constants.%assoc0.fa6]
// CHECK:STDOUT:   %impl.elem0: %.06c = impl_witness_access constants.%Inner.impl_witness.260, element0 [concrete = constants.%C.as.Inner.impl.F.ce3]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.ee2 = acquire_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.f0ba5f.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%c.var)
// CHECK:STDOUT:   return %C.as.Inner.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.2(%self.param: ref %C.ee2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %C => constants.%C.faa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.2c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Self.2c3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Self => constants.%Self.2c3
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.77c
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.d05
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.91b
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.ef7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Self.2c3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Self => constants.%Self.2c3
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%Self.as_type.48e
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.19c
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
// CHECK:STDOUT:   %C => constants.%C.faa
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.539
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.963
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.8c0
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.451
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.faa
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.5c6
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.a2e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Self => constants.%Inner.facet.a2e
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.284
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.493
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.91b
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.ef7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.a2e) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Self => constants.%Inner.facet.a2e
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.faa
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.5c6
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.b31) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Self => constants.%Inner.facet.b31
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.631
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.382
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.91b
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.ef7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.b31) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.549
// CHECK:STDOUT:   %Self => constants.%Inner.facet.b31
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.faa
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.5c6
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %C => constants.%C.ee2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.ec7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Self.2c3) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %Self => constants.%Self.2c3
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.e7b
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.545
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.318
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.fa6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.5cc) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %Self => constants.%Inner.facet.5cc
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.e40
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.792
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.318
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.fa6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.5cc) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %Self => constants.%Inner.facet.5cc
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.522
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.f87
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.17a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.ee2
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.260
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.228
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.44c
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.ce3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.31e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %Self => constants.%Inner.facet.31e
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.d8d
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.e12
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.318
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.fa6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.ee2
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.9e3
// CHECK:STDOUT:   %.loc11_26.2 => constants.%.f87
// CHECK:STDOUT:   %pattern_type.loc11_26 => constants.%pattern_type.17a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_23 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%complete_type.228
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.318
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.fa6
// CHECK:STDOUT:   %.loc11_41.1 => constants.%.d3e
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.260
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.31e
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.d8d
// CHECK:STDOUT:   %.loc11_41.2 => constants.%.06c
// CHECK:STDOUT:   %impl.elem0.loc11_41.2 => constants.%C.as.Inner.impl.F.ce3
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.31e) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.506
// CHECK:STDOUT:   %Self => constants.%Inner.facet.31e
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.ee2
// CHECK:STDOUT:   %pattern_type.loc6_14 => constants.%pattern_type.9e3
// CHECK:STDOUT:   %.loc6_27.1 => constants.%.f87
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.17a
// CHECK:STDOUT: }
// CHECK:STDOUT:
