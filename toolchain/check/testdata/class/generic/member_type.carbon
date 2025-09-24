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
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.be8: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.117: type = class_type @Outer, @Outer(%T.be8) [symbolic]
// CHECK:STDOUT:   %Inner.6ee: type = class_type @Inner, @Inner(%T.be8) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.be8 [symbolic]
// CHECK:STDOUT:   %require_complete.07c: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.elem.86c: type = unbound_element_type %Inner.6ee, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n.d5b: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.934: <witness> = complete_type_witness %struct_type.n.d5b [symbolic]
// CHECK:STDOUT:   %pattern_type.965801.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.a9d: type = pattern_type %Inner.6ee [symbolic]
// CHECK:STDOUT:   %Outer.F.type.f58: type = fn_type @Outer.F, @Outer(%T.be8) [symbolic]
// CHECK:STDOUT:   %Outer.F.5ba: %Outer.F.type.f58 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.f2b: <witness> = require_complete_type %Inner.6ee [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15: <witness> = lookup_impl_witness %T.be8, @Copy [symbolic]
// CHECK:STDOUT:   %.427: type = fn_type_with_self_type %Copy.Op.type, %T.be8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.168: %.427 = impl_witness_access %Copy.lookup_impl_witness.e15, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ce: <specific function> = specific_impl_function %impl.elem0.168, @Copy.Op(%T.be8) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %Outer.3a3: type = class_type @Outer, @Outer(%Copy.facet.c49) [concrete]
// CHECK:STDOUT:   %Inner.d35: type = class_type @Inner, @Inner(%Copy.facet.c49) [concrete]
// CHECK:STDOUT:   %Outer.F.type.f5d: type = fn_type @Outer.F, @Outer(%Copy.facet.c49) [concrete]
// CHECK:STDOUT:   %Outer.F.e06: %Outer.F.type.f5d = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.7f6: type = unbound_element_type %Inner.d35, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.033: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n.033 [concrete]
// CHECK:STDOUT:   %pattern_type.8e8: type = pattern_type %Inner.d35 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d08: <witness> = impl_witness imports.%Copy.impl_witness_table.5a1 [concrete]
// CHECK:STDOUT:   %Copy.facet.8aa: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.d08) [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.e06, @Outer.F(%Copy.facet.c49) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c49 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Inner.d35, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.c43: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.0f4: %AggregateT.as_type.as.Destroy.impl.Op.type.c43 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.44b: type = ptr_type %Inner.d35 [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.0f4, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b02 = import_ref Core//prelude/parts/copy, loc32_31, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.5a1 = impl_witness_table (%Core.import_ref.b02), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.322 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T.be8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc4_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T.be8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_13.1) [symbolic = %Inner (constants.%Inner.6ee)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_13.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.f58)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.f58) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.5ba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.6ee)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.f58) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.5ba)] {
// CHECK:STDOUT:       %n.patt: @Outer.F.%pattern_type.loc9_8 (%pattern_type.965801.1) = binding_pattern n [concrete]
// CHECK:STDOUT:       %n.param_patt: @Outer.F.%pattern_type.loc9_8 (%pattern_type.965801.1) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Outer.F.%pattern_type.loc9_14 (%pattern_type.a9d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Outer.F.%pattern_type.loc9_14 (%pattern_type.a9d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.be8) [symbolic = %Inner (constants.%Inner.6ee)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17 [symbolic = %Inner (constants.%Inner.6ee)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:         %T.as_type.loc9_11.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc9_11.2: type = converted %T.ref, %T.as_type.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = bind_name n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.6ee) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.6ee) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.117
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %T.as_type.loc6_12.2: type = facet_access_type %T [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_12.2 [symbolic = %require_complete (constants.%require_complete.07c)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.6ee)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.as_type.loc6_12.2 [symbolic = %Inner.elem (constants.%Inner.elem.86c)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.as_type.loc6_12.2 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.d5b)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.934)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:     %T.as_type.loc6_12.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6_12: type = converted %T.ref, %T.as_type.loc6_12.1 [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6_10: @Inner.%Inner.elem (%Inner.elem.86c) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.d5b [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.934)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.6ee
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6_10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %T.as_type.loc9_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_8: type = pattern_type %T.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_8 (constants.%pattern_type.965801.1)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.6ee)]
// CHECK:STDOUT:   %pattern_type.loc9_14: type = pattern_type %Inner [symbolic = %pattern_type.loc9_14 (constants.%pattern_type.a9d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.f2b)]
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.as_type.loc9_11.1 [symbolic = %require_complete.loc9_9 (constants.%require_complete.07c)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.d5b)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15)]
// CHECK:STDOUT:   %.loc9_38.2: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc9_38.2 (constants.%.427)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38.2 (%.427) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)) -> %return.param: @Outer.F.%Inner (%Inner.6ee) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.d5b) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38.2 (%.427) = impl_witness_access constants.%Copy.lookup_impl_witness.e15, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.Op(constants.%T.be8) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc9_38.1: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = call %bound_method.loc9_38.2(%n.ref) to %.loc9_39.2
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = initialize_from %.loc9_38.1 to %.loc9_39.2
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.6ee) = class_init (%.loc9_39.3), %return
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.6ee) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.8e8 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.8e8 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.d35 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc13_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.a32) [concrete = constants.%Copy.facet.c49]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.c49]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.c49) [concrete = constants.%Outer.3a3]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.f5d = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.c49) [concrete = constants.%Outer.F.e06]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.f5d = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.e06]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43.1: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.d08) [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %.loc13_43.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.1 [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %Copy.facet.loc13_43.2: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.d08) [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %.loc13_43.2: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.2 [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.c49) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3.1: ref %Inner.d35 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0.loc13: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.d35 = call %Outer.F.specific_fn(%.loc13_42.2) to %.loc13_3.1
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.d35] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc13_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.a32) [concrete = constants.%Copy.facet.c49]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.c49]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.c49) [concrete = constants.%Outer.3a3]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.c49) [concrete = constants.%Inner.d35]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.d35]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.d35 = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.d35 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.7f6 = name_ref n, @Inner.%.loc6_10 [concrete = @Inner.%.loc6_10]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0.loc14: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn.loc14
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%Inner.d35, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc13_3.2: %type_where = converted constants.%Inner.d35, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.0f4
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.0f4, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %c.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.44b = addr_of %c.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_3(%addr)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.be8) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.be8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.6ee
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.f58
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.5ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.be8) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %T.as_type.loc6_12.2 => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.07c
// CHECK:STDOUT:   %Inner => constants.%Inner.6ee
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.86c
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.d5b
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.934
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.be8) {
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.965801.1
// CHECK:STDOUT:   %Inner => constants.%Inner.6ee
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.a9d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.c49) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Copy.facet.c49
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.d35
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.f5d
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.e06
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.c49) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.c49
// CHECK:STDOUT:   %T.as_type.loc6_12.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.d35
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.7f6
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.033
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.54b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.c49) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.c49
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %Inner => constants.%Inner.d35
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.8e8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.54b
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.033
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.a32
// CHECK:STDOUT:   %.loc9_38.2 => constants.%.7fa
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.f59
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.d07: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f03: %Inner.type.d07 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.bd3: type = facet_access_type %Self.f03 [symbolic]
// CHECK:STDOUT:   %pattern_type.120: type = pattern_type %Self.as_type.bd3 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Inner.F.type.0f3: type = fn_type @Inner.F, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.F.db9: %Inner.F.type.0f3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.115: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.95e: %Inner.assoc_type.115 = assoc_entity element0, @Inner.%Inner.F.decl [symbolic]
// CHECK:STDOUT:   %C.390: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.dfd: <witness> = require_complete_type %Inner.type.d07 [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.50b: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e59: type = pattern_type %C.390 [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.d1f: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.51e: %C.as.Inner.impl.F.type.d1f = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.ce6: %Inner.type.d07 = facet_value %C.390, (%Inner.impl_witness.50b) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.930: <witness> = require_complete_type %C.390 [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.390, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.949: %Inner.type.d07 = facet_value %C.390, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.4e7: type = fn_type_with_self_type %Inner.F.type.0f3, %Inner.facet.949 [symbolic]
// CHECK:STDOUT:   %impl.elem0.e4d: %.4e7 = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.3e8: <specific function> = specific_impl_function %impl.elem0.e4d, @Inner.F(%T, %Inner.facet.949) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.545: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.56c: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.70f: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Self.028: %Inner.type.56c = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.F.type.86e: type = fn_type @Inner.F, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %Inner.F.11d: %Inner.F.type.86e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.assoc_type.215: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e80: %Inner.assoc_type.215 = assoc_entity element0, @Inner.%Inner.F.decl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.01d: <witness> = impl_witness @D.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F.type: type = fn_type @D.as.Inner.impl.F [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F: %D.as.Inner.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.d57: %Inner.type.56c = facet_value %D, (%Inner.impl_witness.01d) [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.de9: type = pattern_type %C.70f [concrete]
// CHECK:STDOUT:   %C.val: %C.70f = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.619: <witness> = complete_type_witness %Inner.type.56c [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.828: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.186: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.cd5: %C.as.Inner.impl.F.type.186 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.3aa: %Inner.type.56c = facet_value %C.70f, (%Inner.impl_witness.828) [concrete]
// CHECK:STDOUT:   %.3eb: type = fn_type_with_self_type %Inner.F.type.86e, %Inner.facet.3aa [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.cd5, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C.70f, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.be9: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.46c: %AggregateT.as_type.as.Destroy.impl.Op.type.be9 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.18f: type = ptr_type %C.70f [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.46c, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:   %Self.2: @Inner.%Inner.type (%Inner.type.d07) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.f03)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%T) [symbolic = %Inner.F.type (constants.%Inner.F.type.0f3)]
// CHECK:STDOUT:   %Inner.F: @Inner.%Inner.F.type (%Inner.F.type.0f3) = struct_value () [symbolic = %Inner.F (constants.%Inner.F.db9)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.115)]
// CHECK:STDOUT:   %assoc0.loc6_28.2: @Inner.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, %Inner.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.95e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Inner.%Inner.type (%Inner.type.d07) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.f03)]
// CHECK:STDOUT:     %Inner.F.decl: @Inner.%Inner.F.type (%Inner.F.type.0f3) = fn_decl @Inner.F [symbolic = @Inner.%Inner.F (constants.%Inner.F.db9)] {
// CHECK:STDOUT:       %self.patt: @Inner.F.%pattern_type.loc6_10 (%pattern_type.120) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Inner.F.%pattern_type.loc6_10 (%pattern_type.120) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.F.%pattern_type.loc6_24 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.F.%pattern_type.loc6_24 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Inner.F.%Self.as_type.loc6_16.1 (%Self.as_type.bd3) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %.loc6_16.3 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.bd3)] {
// CHECK:STDOUT:         %.loc6_16.2: @Inner.F.%Inner.type (%Inner.type.d07) = specific_constant @Inner.%Self.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.f03)]
// CHECK:STDOUT:         %Self.ref: @Inner.F.%Inner.type (%Inner.type.d07) = name_ref Self, %.loc6_16.2 [symbolic = %Self (constants.%Self.f03)]
// CHECK:STDOUT:         %Self.as_type.loc6_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.bd3)]
// CHECK:STDOUT:         %.loc6_16.3: type = converted %Self.ref, %Self.as_type.loc6_16.2 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.bd3)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.F.%Self.as_type.loc6_16.1 (%Self.as_type.bd3) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @Inner.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Inner.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_28.1: @Inner.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, %Inner.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.95e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_28.1
// CHECK:STDOUT:     witness = (%Inner.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Inner.impl(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.dfd)]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness (constants.%Inner.impl_witness.50b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.d1f)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.d1f) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.d1f) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.51e)] {
// CHECK:STDOUT:       %self.patt: @C.as.Inner.impl.F.%pattern_type.loc11_12 (%pattern_type.e59) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_12 (%pattern_type.e59) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @C.as.Inner.impl.F.%pattern_type.loc11_23 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_23 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @C.as.Inner.impl.F.%C (%C.390) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.390)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Inner.impl.F.%C (%C.390) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @C.as.Inner.impl.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @C.as.Inner.impl.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %C.as.Inner.impl.F.decl
// CHECK:STDOUT:     .Inner = <poisoned>
// CHECK:STDOUT:     witness = @C.%Inner.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Inner.impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:   %D.as.Inner.impl.F.decl: %D.as.Inner.impl.F.type = fn_decl @D.as.Inner.impl.F [concrete = constants.%D.as.Inner.impl.F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .F = %D.as.Inner.impl.F.decl
// CHECK:STDOUT:   witness = @D.%Inner.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_13.1)> [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_13.1) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.d07)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.390)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @C.as.Inner.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.390 [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (@C.as.Inner.impl.%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T) [symbolic = @C.as.Inner.impl.%Inner.impl_witness (constants.%Inner.impl_witness.50b)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.390
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
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.56c]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.56c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (@D.as.Inner.impl.%D.as.Inner.impl.F.decl), @D.as.Inner.impl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.01d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.F(@Outer.%T.loc4_13.2: type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.d07)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:   %Self: @Inner.F.%Inner.type (%Inner.type.d07) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.f03)]
// CHECK:STDOUT:   %Self.as_type.loc6_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.bd3)]
// CHECK:STDOUT:   %pattern_type.loc6_10: type = pattern_type %Self.as_type.loc6_16.1 [symbolic = %pattern_type.loc6_10 (constants.%pattern_type.120)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %T [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.F.%Self.as_type.loc6_16.1 (%Self.as_type.bd3)) -> %return.param: @Inner.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:   %pattern_type.loc11_12: type = pattern_type %C [symbolic = %pattern_type.loc11_12 (constants.%pattern_type.e59)]
// CHECK:STDOUT:   %pattern_type.loc11_23: type = pattern_type %T [symbolic = %pattern_type.loc11_23 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_23: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.930)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:   %require_complete.loc11_48: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_48 (constants.%require_complete.dfd)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.115)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, @Inner.%Inner.F.decl [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%T) [symbolic = %Inner.F.type (constants.%Inner.F.type.0f3)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.d07) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.949)]
// CHECK:STDOUT:   %.loc11_41: type = fn_type_with_self_type %Inner.F.type, %Inner.facet [symbolic = %.loc11_41 (constants.%.4e7)]
// CHECK:STDOUT:   %impl.elem0.loc11_41.2: @C.as.Inner.impl.F.%.loc11_41 (%.4e7) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0.e4d)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2: <specific function> = specific_impl_function %impl.elem0.loc11_41.2, @Inner.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn.3e8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.390)) -> %return.param: @C.as.Inner.impl.F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.390) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_43: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_43 [symbolic = %Inner.type (constants.%Inner.type.d07)]
// CHECK:STDOUT:     %.loc11_48: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.115) = specific_constant @Inner.%assoc0.loc6_28.1, @Inner(constants.%T) [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.115) = name_ref F, %.loc11_48 [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:     %impl.elem0.loc11_41.1: @C.as.Inner.impl.F.%.loc11_41 (%.4e7) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0.e4d)]
// CHECK:STDOUT:     %bound_method.loc11_41: <bound method> = bound_method %self.ref, %impl.elem0.loc11_41.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_41.1: <specific function> = specific_impl_function %impl.elem0.loc11_41.1, @Inner.F(constants.%T, constants.%Inner.facet.949) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn.3e8)]
// CHECK:STDOUT:     %bound_method.loc11_52: <bound method> = bound_method %self.ref, %specific_impl_fn.loc11_41.1
// CHECK:STDOUT:     %.loc11_23: ref @C.as.Inner.impl.F.%T (%T) = splice_block %return {}
// CHECK:STDOUT:     %.loc11_52: init @C.as.Inner.impl.F.%T (%T) = call %bound_method.loc11_52(%self.ref) to %.loc11_23
// CHECK:STDOUT:     return %.loc11_52 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Inner.impl.F(%self.param: %D) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.de9 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.de9 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.70f = var %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc23_26.2: init %C.70f = class_init (), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_3.1: init %C.70f = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_3.1
// CHECK:STDOUT:   %.loc23_20.1: type = splice_block %C.ref [concrete = constants.%C.70f] {
// CHECK:STDOUT:     %Outer.ref.loc23: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc23: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:     %.loc23_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.70f]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc23_20.2 [concrete = constants.%C.70f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.70f = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.70f = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.56c]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.56c]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.215 = specific_constant @Inner.%assoc0.loc6_28.1, @Inner(constants.%i32) [concrete = constants.%assoc0.e80]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.215 = name_ref F, %.loc24_29 [concrete = constants.%assoc0.e80]
// CHECK:STDOUT:   %impl.elem0: %.3eb = impl_witness_access constants.%Inner.impl_witness.828, element0 [concrete = constants.%C.as.Inner.impl.F.cd5]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.70f = bind_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%C.70f, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc23_3.2: %type_where = converted constants.%C.70f, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.46c
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.46c, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %c.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.18f = addr_of %c.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc23(%addr)
// CHECK:STDOUT:   return %C.as.Inner.impl.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d07
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d07
// CHECK:STDOUT:   %Self.2 => constants.%Self.f03
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.0f3
// CHECK:STDOUT:   %Inner.F => constants.%Inner.F.db9
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.115
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.95e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Self.f03) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d07
// CHECK:STDOUT:   %Self => constants.%Self.f03
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%Self.as_type.bd3
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.120
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d07
// CHECK:STDOUT:   %require_complete => constants.%require_complete.dfd
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.50b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.d1f
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.51e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc11_12 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc11_23 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Inner.facet.ce6) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d07
// CHECK:STDOUT:   %Self => constants.%Inner.facet.ce6
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Inner.facet.949) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.d07
// CHECK:STDOUT:   %Self => constants.%Inner.facet.949
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %Self.2 => constants.%Self.028
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.86e
// CHECK:STDOUT:   %Inner.F => constants.%Inner.F.11d
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.215
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.e80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%i32, constants.%Inner.facet.d57) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %Self => constants.%Inner.facet.d57
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.510
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.619
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.828
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.186
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.cd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT:   %pattern_type.loc11_12 => constants.%pattern_type.de9
// CHECK:STDOUT:   %pattern_type.loc11_23 => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_23 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc11_16 => constants.%complete_type.357
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%complete_type.619
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.215
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e80
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.828
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.86e
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.3aa
// CHECK:STDOUT:   %.loc11_41 => constants.%.3eb
// CHECK:STDOUT:   %impl.elem0.loc11_41.2 => constants.%C.as.Inner.impl.F.cd5
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%i32, constants.%Inner.facet.3aa) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %Self => constants.%Inner.facet.3aa
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.70f
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.de9
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
