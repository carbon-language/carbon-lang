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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.417: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.8c0: type = class_type @Outer, @Outer(%T.417) [symbolic]
// CHECK:STDOUT:   %Inner.bd5: type = class_type @Inner, @Inner(%T.417) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.417 [symbolic]
// CHECK:STDOUT:   %require_complete.d74: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Inner.elem.cdd: type = unbound_element_type %Inner.bd5, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n.a18: type = struct_type {.n: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.088: <witness> = complete_type_witness %struct_type.n.a18 [symbolic]
// CHECK:STDOUT:   %pattern_type.f14b96.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.fc1: type = pattern_type %Inner.bd5 [symbolic]
// CHECK:STDOUT:   %Outer.F.type.ed4: type = fn_type @Outer.F, @Outer(%T.417) [symbolic]
// CHECK:STDOUT:   %Outer.F.280: %Outer.F.type.ed4 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.baa: <witness> = require_complete_type %Inner.bd5 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.c42: <witness> = lookup_impl_witness %T.417, @Copy [symbolic]
// CHECK:STDOUT:   %.a79: type = fn_type_with_self_type %Copy.Op.type, %T.417 [symbolic]
// CHECK:STDOUT:   %impl.elem0.fac: %.a79 = impl_witness_access %Copy.lookup_impl_witness.c42, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.103: <specific function> = specific_impl_function %impl.elem0.fac, @Copy.Op(%T.417) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c50: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %Outer.4f9: type = class_type @Outer, @Outer(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %Inner.154: type = class_type @Inner, @Inner(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %Outer.F.type.0f4: type = fn_type @Outer.F, @Outer(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %Outer.F.91d: %Outer.F.type.0f4 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.d06: type = unbound_element_type %Inner.154, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.033: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n.033 [concrete]
// CHECK:STDOUT:   %pattern_type.15f: type = pattern_type %Inner.154 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d08: <witness> = impl_witness imports.%Copy.impl_witness_table.5a1 [concrete]
// CHECK:STDOUT:   %Copy.facet.8aa: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.d08) [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.91d, @Outer.F(%Copy.facet.c50) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c50 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %Inner.154, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba9: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.54b: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.ba9 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.54b, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b02 = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.5a1 = impl_witness_table (%Core.import_ref.b02), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_13.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T.417)]
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
// CHECK:STDOUT:   %T.loc4_13.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T.417)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_13.1) [symbolic = %Inner (constants.%Inner.bd5)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_13.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.ed4)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.ed4) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.280)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.bd5)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.ed4) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.280)] {
// CHECK:STDOUT:       %n.patt: @Outer.F.%pattern_type.loc9_8 (%pattern_type.f14b96.1) = value_binding_pattern n [concrete]
// CHECK:STDOUT:       %n.param_patt: @Outer.F.%pattern_type.loc9_8 (%pattern_type.f14b96.1) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Outer.F.%pattern_type.loc9_14 (%pattern_type.fc1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Outer.F.%pattern_type.loc9_14 (%pattern_type.fc1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.417) [symbolic = %Inner (constants.%Inner.bd5)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17 [symbolic = %Inner (constants.%Inner.bd5)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.417)]
// CHECK:STDOUT:         %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:         %.loc9_11.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.binding.as_type (%T.binding.as_type) = value_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.bd5) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.bd5) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.8c0
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.417)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.d74)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.bd5)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.binding.as_type [symbolic = %Inner.elem (constants.%Inner.elem.cdd)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.binding.as_type (%T.binding.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.a18)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.088)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.417)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc6_12: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc6_10: @Inner.%Inner.elem (%Inner.elem.cdd) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.a18 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.088)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.bd5
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6_10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.417)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_8: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9_8 (constants.%pattern_type.f14b96.1)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.bd5)]
// CHECK:STDOUT:   %pattern_type.loc9_14: type = pattern_type %Inner [symbolic = %pattern_type.loc9_14 (constants.%pattern_type.fc1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.baa)]
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9_9 (constants.%require_complete.d74)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.binding.as_type (%T.binding.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.a18)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.c42)]
// CHECK:STDOUT:   %.loc9_38.2: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc9_38.2 (constants.%.a79)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38.2 (%.a79) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.fac)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.103)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.binding.as_type (%T.binding.as_type)) -> %return.param: @Outer.F.%Inner (%Inner.bd5) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.binding.as_type (%T.binding.as_type) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.a18) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38.2 (%.a79) = impl_witness_access constants.%Copy.lookup_impl_witness.c42, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.fac)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.Op(constants.%T.417) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.103)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.binding.as_type (%T.binding.as_type) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc9_38.1: init @Outer.F.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc9_38.2(%n.ref) to %.loc9_39.2
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.binding.as_type (%T.binding.as_type) = initialize_from %.loc9_38.1 to %.loc9_39.2
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.bd5) = class_init (%.loc9_39.3), %return
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.bd5) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.15f = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.15f = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.154 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc13_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value %i32.loc13_35, (constants.%Copy.impl_witness.fb7) [concrete = constants.%Copy.facet.c50]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.c50]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.c50) [concrete = constants.%Outer.4f9]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.0f4 = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.c50) [concrete = constants.%Outer.F.91d]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.0f4 = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.91d]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43.1: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.d08) [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %.loc13_43.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.1 [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %Copy.facet.loc13_43.2: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.d08) [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %.loc13_43.2: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.2 [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.c50) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3: ref %Inner.154 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0.loc13: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.154 = call %Outer.F.specific_fn(%.loc13_42.2) to %.loc13_3
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.154] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc13_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value %i32.loc13_16, (constants.%Copy.impl_witness.fb7) [concrete = constants.%Copy.facet.c50]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.c50]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.c50) [concrete = constants.%Outer.4f9]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.c50) [concrete = constants.%Inner.154]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.154]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.154 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.154 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.d06 = name_ref n, @Inner.%.loc6_10 [concrete = @Inner.%.loc6_10]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = acquire_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0.loc14: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn.loc14
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.54b
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.54b, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_3(%c.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.417) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.417
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.bd5
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.ed4
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.417) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.417
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d74
// CHECK:STDOUT:   %Inner => constants.%Inner.bd5
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.cdd
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.a18
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.088
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.417) {
// CHECK:STDOUT:   %T => constants.%T.417
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.f14b96.1
// CHECK:STDOUT:   %Inner => constants.%Inner.bd5
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.fc1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.c50) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Copy.facet.c50
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.154
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.0f4
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.91d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.c50) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.c50
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.154
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.d06
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.033
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.54b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.c50) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.c50
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %Inner => constants.%Inner.154
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.15f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.54b
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.033
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.fb7
// CHECK:STDOUT:   %.loc9_38.2 => constants.%.65f
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.dfd
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.d8f: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.5f4: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1ac: %Inner.type.5f4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type.bfd: type = symbolic_binding_type Self, 1, %Self.1ac [symbolic]
// CHECK:STDOUT:   %pattern_type.0d2: type = pattern_type %Self.binding.as_type.bfd [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Inner.F.type.b25: type = fn_type @Inner.F, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.F.88e: %Inner.F.type.b25 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.b54: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.dc7: %Inner.assoc_type.b54 = assoc_entity element0, @Inner.%Inner.F.decl [symbolic]
// CHECK:STDOUT:   %C.52d: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.3ba: <witness> = require_complete_type %Inner.type.5f4 [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.9fc: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2d3: type = pattern_type %C.52d [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.2a9: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.bf7: %C.as.Inner.impl.F.type.2a9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.64b: %Inner.type.5f4 = facet_value %C.52d, (%Inner.impl_witness.9fc) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.a8f: <witness> = require_complete_type %C.52d [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.52d, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.318: %Inner.type.5f4 = facet_value %C.52d, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.b0e: type = fn_type_with_self_type %Inner.F.type.b25, %Inner.facet.318 [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b0e = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Inner.F(%T, %Inner.facet.318) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.545: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.56c: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.70f: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Self.d26: %Inner.type.56c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.F.type.86e: type = fn_type @Inner.F, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %Inner.F.11d: %Inner.F.type.86e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.assoc_type.215: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.eba: %Inner.assoc_type.215 = assoc_entity element0, @Inner.%Inner.F.decl [concrete]
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
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C.70f = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.619: <witness> = complete_type_witness %Inner.type.56c [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.f06: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.aa4: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.709: %C.as.Inner.impl.F.type.aa4 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.98e: %Inner.type.56c = facet_value %C.70f, (%Inner.impl_witness.f06) [concrete]
// CHECK:STDOUT:   %.3cd: type = fn_type_with_self_type %Inner.F.type.86e, %Inner.facet.98e [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.709, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %C.70f, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cbf: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.191: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.cbf = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.191, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:   %Self.loc5_19.2: @Inner.%Inner.type (%Inner.type.5f4) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.1ac)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%T) [symbolic = %Inner.F.type (constants.%Inner.F.type.b25)]
// CHECK:STDOUT:   %Inner.F: @Inner.%Inner.F.type (%Inner.F.type.b25) = struct_value () [symbolic = %Inner.F (constants.%Inner.F.88e)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.b54)]
// CHECK:STDOUT:   %assoc0.loc6_28.2: @Inner.%Inner.assoc_type (%Inner.assoc_type.b54) = assoc_entity element0, %Inner.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.dc7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.5f4) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.1ac)]
// CHECK:STDOUT:     %Inner.F.decl: @Inner.%Inner.F.type (%Inner.F.type.b25) = fn_decl @Inner.F [symbolic = @Inner.%Inner.F (constants.%Inner.F.88e)] {
// CHECK:STDOUT:       %self.patt: @Inner.F.%pattern_type.loc6_10 (%pattern_type.0d2) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Inner.F.%pattern_type.loc6_10 (%pattern_type.0d2) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.F.%pattern_type.loc6_24 (%pattern_type.e68) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.F.%pattern_type.loc6_24 (%pattern_type.e68) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Inner.F.%Self.binding.as_type (%Self.binding.as_type.bfd) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %.loc6_16.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bfd)] {
// CHECK:STDOUT:         %.loc6_16.2: @Inner.F.%Inner.type (%Inner.type.5f4) = specific_constant @Inner.%Self.loc5_19.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.1ac)]
// CHECK:STDOUT:         %Self.ref: @Inner.F.%Inner.type (%Inner.type.5f4) = name_ref Self, %.loc6_16.2 [symbolic = %Self (constants.%Self.1ac)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bfd)]
// CHECK:STDOUT:         %.loc6_16.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bfd)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.F.%Self.binding.as_type (%Self.binding.as_type.bfd) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Inner.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Inner.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_28.1: @Inner.%Inner.assoc_type (%Inner.assoc_type.b54) = assoc_entity element0, %Inner.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.dc7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_19.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_28.1
// CHECK:STDOUT:     witness = (%Inner.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @C.as.Inner.impl(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.52d)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.3ba)]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness (constants.%Inner.impl_witness.9fc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.2a9)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.2a9) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.bf7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.2a9) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.bf7)] {
// CHECK:STDOUT:       %self.patt: @C.as.Inner.impl.F.%pattern_type.loc11_12 (%pattern_type.2d3) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_12 (%pattern_type.2d3) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @C.as.Inner.impl.F.%pattern_type.loc11_23 (%pattern_type.e68) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_23 (%pattern_type.e68) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @C.as.Inner.impl.F.%C (%C.52d) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.52d)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T) [symbolic = %C (constants.%C.52d)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.52d)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Inner.impl.F.%C (%C.52d) = value_binding self, %self.param
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
// CHECK:STDOUT:     %self.patt: %pattern_type.510 = value_binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.510 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
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
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_13.1)> [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_13.1) [symbolic = %C (constants.%C.52d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.5f4)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.52d)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.d8f
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
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.52d [symbolic = %C (constants.%C.52d)]
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (@C.as.Inner.impl.%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T) [symbolic = @C.as.Inner.impl.%Inner.impl_witness (constants.%Inner.impl_witness.9fc)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.52d
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
// CHECK:STDOUT: generic fn @Inner.F(@Outer.%T.loc4_13.2: type, @Inner.%Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.5f4)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:   %Self: @Inner.F.%Inner.type (%Inner.type.5f4) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.1ac)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type.bfd)]
// CHECK:STDOUT:   %pattern_type.loc6_10: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc6_10 (constants.%pattern_type.0d2)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %T [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.F.%Self.binding.as_type (%Self.binding.as_type.bfd)) -> %return.param: @Inner.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.52d)]
// CHECK:STDOUT:   %pattern_type.loc11_12: type = pattern_type %C [symbolic = %pattern_type.loc11_12 (constants.%pattern_type.2d3)]
// CHECK:STDOUT:   %pattern_type.loc11_23: type = pattern_type %T [symbolic = %pattern_type.loc11_23 (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_23: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_23 (constants.%require_complete.4b7)]
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.a8f)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:   %require_complete.loc11_48: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_48 (constants.%require_complete.3ba)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.b54)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.b54) = assoc_entity element0, @Inner.%Inner.F.decl [symbolic = %assoc0 (constants.%assoc0.dc7)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%T) [symbolic = %Inner.F.type (constants.%Inner.F.type.b25)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.5f4) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.318)]
// CHECK:STDOUT:   %.loc11_41: type = fn_type_with_self_type %Inner.F.type, %Inner.facet [symbolic = %.loc11_41 (constants.%.b0e)]
// CHECK:STDOUT:   %impl.elem0.loc11_41.2: @C.as.Inner.impl.F.%.loc11_41 (%.b0e) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2: <specific function> = specific_impl_function %impl.elem0.loc11_41.2, @Inner.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.52d)) -> %return.param: @C.as.Inner.impl.F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.52d) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_43: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_43 [symbolic = %Inner.type (constants.%Inner.type.5f4)]
// CHECK:STDOUT:     %.loc11_48: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.b54) = specific_constant @Inner.%assoc0.loc6_28.1, @Inner(constants.%T) [symbolic = %assoc0 (constants.%assoc0.dc7)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.b54) = name_ref F, %.loc11_48 [symbolic = %assoc0 (constants.%assoc0.dc7)]
// CHECK:STDOUT:     %impl.elem0.loc11_41.1: @C.as.Inner.impl.F.%.loc11_41 (%.b0e) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc11_41: <bound method> = bound_method %self.ref, %impl.elem0.loc11_41.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_41.1: <specific function> = specific_impl_function %impl.elem0.loc11_41.1, @Inner.F(constants.%T, constants.%Inner.facet.318) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
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
// CHECK:STDOUT:     %c.patt: %pattern_type.de9 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.de9 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.70f = var %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_26.2: init %C.70f = class_init (), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_3: init %C.70f = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_3
// CHECK:STDOUT:   %.loc23_20.1: type = splice_block %C.ref [concrete = constants.%C.70f] {
// CHECK:STDOUT:     %Outer.ref.loc23: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc23: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc23: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc23: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:     %.loc23_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.70f]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc23_20.2 [concrete = constants.%C.70f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.70f = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.70f = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.56c]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.56c]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.215 = specific_constant @Inner.%assoc0.loc6_28.1, @Inner(constants.%i32) [concrete = constants.%assoc0.eba]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.215 = name_ref F, %.loc24_29 [concrete = constants.%assoc0.eba]
// CHECK:STDOUT:   %impl.elem0: %.3cd = impl_witness_access constants.%Inner.impl_witness.f06, element0 [concrete = constants.%C.as.Inner.impl.F.709]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.70f = acquire_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.191
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.191, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %c.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc23(%c.var)
// CHECK:STDOUT:   return %C.as.Inner.impl.F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5f4
// CHECK:STDOUT:   %C => constants.%C.52d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5f4
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.1ac
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.b25
// CHECK:STDOUT:   %Inner.F => constants.%Inner.F.88e
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.b54
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.dc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Self.1ac) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5f4
// CHECK:STDOUT:   %Self => constants.%Self.1ac
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type.bfd
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.0d2
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.52d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5f4
// CHECK:STDOUT:   %require_complete => constants.%require_complete.3ba
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.9fc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.2a9
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.bf7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.52d
// CHECK:STDOUT:   %pattern_type.loc11_12 => constants.%pattern_type.2d3
// CHECK:STDOUT:   %pattern_type.loc11_23 => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Inner.facet.64b) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5f4
// CHECK:STDOUT:   %Self => constants.%Inner.facet.64b
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C.52d
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.2d3
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.e68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Inner.facet.318) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5f4
// CHECK:STDOUT:   %Self => constants.%Inner.facet.318
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C.52d
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.2d3
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.e68
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
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.d26
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.86e
// CHECK:STDOUT:   %Inner.F => constants.%Inner.F.11d
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.215
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.eba
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
// CHECK:STDOUT:   %Self.binding.as_type => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.510
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %require_complete => constants.%complete_type.619
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.f06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.aa4
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.709
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
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.eba
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.f06
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.86e
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.98e
// CHECK:STDOUT:   %.loc11_41 => constants.%.3cd
// CHECK:STDOUT:   %impl.elem0.loc11_41.2 => constants.%C.as.Inner.impl.F.709
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%i32, constants.%Inner.facet.98e) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.56c
// CHECK:STDOUT:   %Self => constants.%Inner.facet.98e
// CHECK:STDOUT:   %Self.binding.as_type => constants.%C.70f
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.de9
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
