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
    fn F(self) -> T;
  }

  class C {
    impl as Inner {
      fn F(self: C) -> T { return self.(Inner.F)(); }
    }
  }
}

class D {
  impl as Outer(i32).Inner {
    fn F(self: D) -> i32;
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
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.409: type = class_type @Outer, @Outer(%T.f84) [symbolic]
// CHECK:STDOUT:   %Inner.06b: type = class_type @Inner, @Inner(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %n.patt.872: %pattern_type.2c66b4.2 = ref_binding_pattern n [symbolic]
// CHECK:STDOUT:   %Inner.elem.51d: type = unbound_element_type %Inner.06b, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n.797: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.22c: <witness> = complete_type_witness %struct_type.n.797 [symbolic]
// CHECK:STDOUT:   %n.param_patt.18f: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %n.patt.8c9: %pattern_type.2c66b4.2 = at_binding_pattern n, %n.param_patt.18f [symbolic]
// CHECK:STDOUT:   %.c17: Core.Form = init_form %Inner.06b [symbolic]
// CHECK:STDOUT:   %pattern_type.c1c: type = pattern_type %Inner.06b [symbolic]
// CHECK:STDOUT:   %return.param_patt.953: %pattern_type.c1c = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.0b4: %pattern_type.c1c = return_slot_pattern %return.param_patt.953, %Inner.06b [symbolic]
// CHECK:STDOUT:   %Outer.F.type.823: type = fn_type @Outer.F, @Outer(%T.f84) [symbolic]
// CHECK:STDOUT:   %Outer.F.e27: %Outer.F.type.823 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.972: <witness> = require_complete_type %Inner.06b [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.987: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Outer.90e: type = class_type @Outer, @Outer(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %Inner.096: type = class_type @Inner, @Inner(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %Outer.F.type.3ad: type = fn_type @Outer.F, @Outer(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %Outer.F.17f: %Outer.F.type.3ad = struct_value () [concrete]
// CHECK:STDOUT:   %n.patt.e55: %pattern_type.6b6 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %Inner.elem.362: type = unbound_element_type %Inner.096, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.9ed: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n.9ed [concrete]
// CHECK:STDOUT:   %pattern_type.1e8: type = pattern_type %Inner.096 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.1e8 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.1e8 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6da: <witness> = impl_witness imports.%Copy.impl_witness_table.06e [concrete]
// CHECK:STDOUT:   %Copy.facet.f12: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.6da) [concrete]
// CHECK:STDOUT:   %n.param_patt.47b: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt.f0b: %pattern_type.6b6 = at_binding_pattern n, %n.param_patt.47b [concrete]
// CHECK:STDOUT:   %.8bc: Core.Form = init_form %Inner.096 [concrete]
// CHECK:STDOUT:   %return.param_patt.3fe: %pattern_type.1e8 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.574: %pattern_type.1e8 = return_slot_pattern %return.param_patt.3fe, %Inner.096 [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.17f, @Outer.F(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet.987 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc13_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bbf = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.06e = impl_witness_table (%Core.import_ref.bbf), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.f84)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc12 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32.loc12 [concrete = constants.%.795]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc4_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.f84)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_14.1) [symbolic = %Inner (constants.%Inner.06b)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_14.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.823)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.823) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.e27)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.06b)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.823) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.e27)] {
// CHECK:STDOUT:       %n.param_patt.loc9_9.1: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %n.param_patt.loc9_9.2 (constants.%n.param_patt.18f)]
// CHECK:STDOUT:       %n.patt.loc9_9.1: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = at_binding_pattern n, %n.param_patt.loc9_9.1 [symbolic = %n.patt.loc9_9.2 (constants.%n.patt.8c9)]
// CHECK:STDOUT:       %return.param_patt.loc9_17.1: @Outer.F.%pattern_type.loc9_17 (%pattern_type.c1c) = out_param_pattern [symbolic = %return.param_patt.loc9_17.2 (constants.%return.param_patt.953)]
// CHECK:STDOUT:       %return.patt.loc9_14.1: @Outer.F.%pattern_type.loc9_17 (%pattern_type.c1c) = return_slot_pattern %return.param_patt.loc9_17.1, %Inner.ref [symbolic = %return.patt.loc9_14.2 (constants.%return.patt.0b4)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.f84) [symbolic = %Inner (constants.%Inner.06b)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17.2 [symbolic = %Inner (constants.%Inner.06b)]
// CHECK:STDOUT:       %.loc9_17.3: Core.Form = init_form %Inner.ref [symbolic = %.loc9_17.1 (constants.%.c17)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.as_type.loc9_11.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc9_11.2: type = converted %T.ref, %T.as_type.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.06b) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.06b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.409
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %n.patt: @Inner.%pattern_type (%pattern_type.2c66b4.2) = ref_binding_pattern n [symbolic = %n.patt (constants.%n.patt.872)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.06b)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.as_type [symbolic = %Inner.elem (constants.%Inner.elem.51d)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.as_type (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.797)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.22c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6: @Inner.%Inner.elem (%Inner.elem.51d) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.797 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.22c)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.06b
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc9_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_9: type = pattern_type %T.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_9 (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %n.param_patt.loc9_9.2: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %n.param_patt.loc9_9.2 (constants.%n.param_patt.18f)]
// CHECK:STDOUT:   %n.patt.loc9_9.2: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = at_binding_pattern n, %n.param_patt.loc9_9.2 [symbolic = %n.patt.loc9_9.2 (constants.%n.patt.8c9)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.06b)]
// CHECK:STDOUT:   %.loc9_17.1: Core.Form = init_form %Inner [symbolic = %.loc9_17.1 (constants.%.c17)]
// CHECK:STDOUT:   %pattern_type.loc9_17: type = pattern_type %Inner [symbolic = %pattern_type.loc9_17 (constants.%pattern_type.c1c)]
// CHECK:STDOUT:   %return.param_patt.loc9_17.2: @Outer.F.%pattern_type.loc9_17 (%pattern_type.c1c) = out_param_pattern [symbolic = %return.param_patt.loc9_17.2 (constants.%return.param_patt.953)]
// CHECK:STDOUT:   %return.patt.loc9_14.2: @Outer.F.%pattern_type.loc9_17 (%pattern_type.c1c) = return_slot_pattern %return.param_patt.loc9_17.2, %Inner [symbolic = %return.patt.loc9_14.2 (constants.%return.patt.0b4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.as_type.loc9_11.1 [symbolic = %require_complete.loc9_9 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.972)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.797)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc9_38.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_38.3 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38.3 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)) -> out %return.param: @Outer.F.%Inner (%Inner.06b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.797) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38.3 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %.loc9_38.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc9_38.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = class_element_access %return.param, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = call %bound_method.loc9_38.2(%n.ref)
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) to %.loc9_39.2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.06b) to %return.param = class_init (%.loc9_39.3)
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.06b) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.1e8 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.1e8 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.096 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc13_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value %i32.loc13_35, (constants.%Copy.impl_witness.0e0) [concrete = constants.%Copy.facet.987]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.987]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.987) [concrete = constants.%Outer.90e]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.3ad = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.987) [concrete = constants.%Outer.F.17f]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.3ad = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.17f]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43.1: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.6da) [concrete = constants.%Copy.facet.f12]
// CHECK:STDOUT:   %.loc13_43.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.1 [concrete = constants.%Copy.facet.f12]
// CHECK:STDOUT:   %Copy.facet.loc13_43.2: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.6da) [concrete = constants.%Copy.facet.f12]
// CHECK:STDOUT:   %.loc13_43.2: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.2 [concrete = constants.%Copy.facet.f12]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.987) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3: ref %Inner.096 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0.loc13: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.096 to %.loc13_3 = call %Outer.F.specific_fn(%.loc13_42.2)
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.096] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc13_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value %i32.loc13_16, (constants.%Copy.impl_witness.0e0) [concrete = constants.%Copy.facet.987]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.987]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.987) [concrete = constants.%Outer.90e]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.987) [concrete = constants.%Inner.096]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.096]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.096 = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.096 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.362 = name_ref n, @Inner.%.loc6 [concrete = @Inner.%.loc6]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = acquire_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0.loc14: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn.loc14
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.1a2547.4
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
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.3(%self.param: ref %struct_type.n.9ed) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_3.4(%self.param: ref %Inner.096) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.f84
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.06b
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.823
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.e27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.f84) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %n.patt => constants.%n.patt.872
// CHECK:STDOUT:   %Inner => constants.%Inner.06b
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.51d
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.797
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.22c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %n.param_patt.loc9_9.2 => constants.%n.param_patt.18f
// CHECK:STDOUT:   %n.patt.loc9_9.2 => constants.%n.patt.8c9
// CHECK:STDOUT:   %Inner => constants.%Inner.06b
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.c17
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.c1c
// CHECK:STDOUT:   %return.param_patt.loc9_17.2 => constants.%return.param_patt.953
// CHECK:STDOUT:   %return.patt.loc9_14.2 => constants.%return.patt.0b4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.987) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.987
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.096
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.3ad
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.17f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.987) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.987
// CHECK:STDOUT:   %T.as_type => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %n.patt => constants.%n.patt.e55
// CHECK:STDOUT:   %Inner => constants.%Inner.096
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.362
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.9ed
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.cdf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.987) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.987
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %n.param_patt.loc9_9.2 => constants.%n.param_patt.47b
// CHECK:STDOUT:   %n.patt.loc9_9.2 => constants.%n.patt.f0b
// CHECK:STDOUT:   %Inner => constants.%Inner.096
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.8bc
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.1e8
// CHECK:STDOUT:   %return.param_patt.loc9_17.2 => constants.%return.param_patt.3fe
// CHECK:STDOUT:   %return.patt.loc9_14.2 => constants.%return.patt.574
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.cdf
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.9ed
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d09
// CHECK:STDOUT:   %.loc9_38.3 => constants.%.7a6
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.0e0
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.4f6
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.dee: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.84b: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.656: %Inner.type.84b = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.abe: type = facet_access_type %Self.656 [symbolic]
// CHECK:STDOUT:   %pattern_type.397: type = pattern_type %Self.as_type.abe [symbolic]
// CHECK:STDOUT:   %self.param_patt.1f9: %pattern_type.397 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.621: %pattern_type.397 = at_binding_pattern self, %self.param_patt.1f9 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.792: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Self.656) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.576: %Inner.WithSelf.F.type.792 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.b6b: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.6dc: %Inner.assoc_type.b6b = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %C.7d7: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.d0a: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.fcc: <witness> = require_complete_type %Inner.type.84b [symbolic]
// CHECK:STDOUT:   %pattern_type.6fd: type = pattern_type %C.7d7 [symbolic]
// CHECK:STDOUT:   %self.param_patt.b1b: %pattern_type.6fd = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.b71: %pattern_type.6fd = at_binding_pattern self, %self.param_patt.b1b [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.1ec: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.4be: %C.as.Inner.impl.F.type.1ec = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.285: %Inner.type.84b = facet_value %C.7d7, (%Inner.impl_witness.d0a) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.b47: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.285) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.350: %Inner.WithSelf.F.type.b47 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.895: <witness> = require_complete_type %C.7d7 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.7d7, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %.72e: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.8ef: %Inner.type.84b = facet_value %C.7d7, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.26a: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet.8ef) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.3e0: %Inner.WithSelf.F.type.26a = struct_value () [symbolic]
// CHECK:STDOUT:   %.35d: type = fn_type_with_self_type %Inner.WithSelf.F.type.26a, %Inner.facet.8ef [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.35d = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @Inner.WithSelf.F(%T, %Inner.facet.8ef) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.e7c: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.5cc: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.b7f: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.8fa: <witness> = impl_witness @D.as.Inner.impl.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.8b6: %Inner.type.5cc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.62d: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Self.656) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.389: %Inner.WithSelf.F.type.62d = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.477: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.7b8: %Inner.assoc_type.477 = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %self.param_patt.f16: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6c0: %pattern_type.d8d = at_binding_pattern self, %self.param_patt.f16 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F.type: type = fn_type @D.as.Inner.impl.F [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F: %D.as.Inner.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.45b: %Inner.type.5cc = facet_value %D, (%Inner.impl_witness.8fa) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.576: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.45b) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.adb: %Inner.WithSelf.F.type.576 = struct_value () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9d0: type = pattern_type %C.b7f [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.9d0 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.9d0 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C.b7f = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.cdc: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.11e: <witness> = complete_type_witness %Inner.type.5cc [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.237: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.e77: %C.as.Inner.impl.F.type.237 = struct_value () [concrete]
// CHECK:STDOUT:   %.fff: require_specific_def_type = require_specific_def @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %Inner.facet.39b: %Inner.type.5cc = facet_value %C.b7f, (%Inner.impl_witness.cdc) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.a78: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.39b) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.e60: %Inner.WithSelf.F.type.a78 = struct_value () [concrete]
// CHECK:STDOUT:   %.8da: type = fn_type_with_self_type %Inner.WithSelf.F.type.a78, %Inner.facet.39b [concrete]
// CHECK:STDOUT:   %self.param_patt.edf: %pattern_type.9d0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.14b: %pattern_type.9d0 = at_binding_pattern self, %self.param_patt.edf [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.e77, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc23_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc22 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22: Core.Form = init_form %i32.loc22 [concrete = constants.%.795]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:   %Self.loc5_19.2: @Inner.%Inner.type (%Inner.type.84b) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.656)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.84b) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.656)]
// CHECK:STDOUT:     %Inner.WithSelf.decl = interface_with_self_decl @Inner [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Inner.WithSelf.F.decl: @Inner.WithSelf.%Inner.WithSelf.F.type (%Inner.WithSelf.F.type.792) = fn_decl @Inner.WithSelf.F [symbolic = @Inner.WithSelf.%Inner.WithSelf.F (constants.%Inner.WithSelf.F.576)] {
// CHECK:STDOUT:       %self.param_patt.loc6_10.1: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.397) = value_param_pattern [symbolic = %self.param_patt.loc6_10.2 (constants.%self.param_patt.1f9)]
// CHECK:STDOUT:       %self.patt.loc6_10.1: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.397) = at_binding_pattern self, %self.param_patt.loc6_10.1 [symbolic = %self.patt.loc6_10.2 (constants.%self.patt.621)]
// CHECK:STDOUT:       %return.param_patt.loc6_19.1: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_19.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc6_16.1: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_19.1, %T.ref [symbolic = %return.patt.loc6_16.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc6_19.2: Core.Form = init_form %T.ref [symbolic = %.loc6_19.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Inner.WithSelf.F.%Self.as_type.loc6_10.1 (%Self.as_type.abe) = value_param call_param0
// CHECK:STDOUT:       %.loc6_10.1: type = splice_block %.loc6_10.3 [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.abe)] {
// CHECK:STDOUT:         %.loc6_10.2: @Inner.WithSelf.F.%Inner.type (%Inner.type.84b) = specific_constant @Inner.%Self.loc5_19.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.656)]
// CHECK:STDOUT:         %Self.ref: @Inner.WithSelf.F.%Inner.type (%Inner.type.84b) = name_ref Self, %.loc6_10.2 [symbolic = %Self (constants.%Self.656)]
// CHECK:STDOUT:         %Self.as_type.loc6_10.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.abe)]
// CHECK:STDOUT:         %.loc6_10.3: type = converted %Self.ref, %Self.as_type.loc6_10.2 [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.abe)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.WithSelf.F.%Self.as_type.loc6_10.1 (%Self.as_type.abe) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Inner.WithSelf.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Inner.WithSelf.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_20.1: @Inner.WithSelf.%Inner.assoc_type (%Inner.assoc_type.b6b) = assoc_entity element0, %Inner.WithSelf.F.decl [symbolic = %assoc0.loc6_20.2 (constants.%assoc0.6dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc5_19.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @Inner.WithSelf.%assoc0.loc6_20.1
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.7d7)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.d0a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.fcc)]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.1ec)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.1ec) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.4be)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.1ec) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.4be)] {
// CHECK:STDOUT:       %self.param_patt.loc11_16.1: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.6fd) = value_param_pattern [symbolic = %self.param_patt.loc11_16.2 (constants.%self.param_patt.b1b)]
// CHECK:STDOUT:       %self.patt.loc11_16.1: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.6fd) = at_binding_pattern self, %self.param_patt.loc11_16.1 [symbolic = %self.patt.loc11_16.2 (constants.%self.patt.b71)]
// CHECK:STDOUT:       %return.param_patt.loc11_24.1: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc11_24.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc11_21.1: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc11_24.1, %T.ref [symbolic = %return.patt.loc11_21.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc11_24.3: Core.Form = init_form %T.ref [symbolic = %.loc11_24.2 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @C.as.Inner.impl.F.%C (%C.7d7) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.7d7)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T) [symbolic = %C (constants.%C.7d7)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.7d7)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Inner.impl.F.%C (%C.7d7) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @C.as.Inner.impl.F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @C.as.Inner.impl.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness.loc10_19.1: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.d0a)]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = value_param_pattern [concrete = constants.%self.param_patt.f16]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6c0]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (%D.as.Inner.impl.F.decl), @D.as.Inner.impl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.8fa]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .F = %D.as.Inner.impl.F.decl
// CHECK:STDOUT:   extend %Inner.ref
// CHECK:STDOUT:   witness = %Inner.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_14.1)> [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_14.1) [symbolic = %C (constants.%C.7d7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.84b)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.7d7)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.dee
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
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.7d7 [symbolic = %C (constants.%C.7d7)]
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.7d7
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
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.e7c]
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.5cc]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.5cc]
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
// CHECK:STDOUT: generic fn @Inner.WithSelf.F(@Outer.%T.loc4_14.2: type, @Inner.%Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.84b)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:   %Self: @Inner.WithSelf.F.%Inner.type (%Inner.type.84b) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.656)]
// CHECK:STDOUT:   %Self.as_type.loc6_10.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.abe)]
// CHECK:STDOUT:   %pattern_type.loc6_10: type = pattern_type %Self.as_type.loc6_10.1 [symbolic = %pattern_type.loc6_10 (constants.%pattern_type.397)]
// CHECK:STDOUT:   %self.param_patt.loc6_10.2: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.397) = value_param_pattern [symbolic = %self.param_patt.loc6_10.2 (constants.%self.param_patt.1f9)]
// CHECK:STDOUT:   %self.patt.loc6_10.2: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.397) = at_binding_pattern self, %self.param_patt.loc6_10.2 [symbolic = %self.patt.loc6_10.2 (constants.%self.patt.621)]
// CHECK:STDOUT:   %.loc6_19.1: Core.Form = init_form %T [symbolic = %.loc6_19.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %T [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc6_19.2: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_19.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc6_16.2: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_19.2, %T [symbolic = %return.patt.loc6_16.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.WithSelf.F.%Self.as_type.loc6_10.1 (%Self.as_type.abe)) -> out %return.param: @Inner.WithSelf.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.7d7)]
// CHECK:STDOUT:   %pattern_type.loc11_16: type = pattern_type %C [symbolic = %pattern_type.loc11_16 (constants.%pattern_type.6fd)]
// CHECK:STDOUT:   %self.param_patt.loc11_16.2: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.6fd) = value_param_pattern [symbolic = %self.param_patt.loc11_16.2 (constants.%self.param_patt.b1b)]
// CHECK:STDOUT:   %self.patt.loc11_16.2: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.6fd) = at_binding_pattern self, %self.param_patt.loc11_16.2 [symbolic = %self.patt.loc11_16.2 (constants.%self.patt.b71)]
// CHECK:STDOUT:   %.loc11_24.2: Core.Form = init_form %T [symbolic = %.loc11_24.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc11_24: type = pattern_type %T [symbolic = %pattern_type.loc11_24 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc11_24.2: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc11_24.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc11_21.2: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc11_24.2, %T [symbolic = %return.patt.loc11_21.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.895)]
// CHECK:STDOUT:   %require_complete.loc11_21: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_21 (constants.%require_complete.944)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:   %require_complete.loc11_46: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_46 (constants.%require_complete.fcc)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.b6b)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.b6b) = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.6dc)]
// CHECK:STDOUT:   %.loc11_39.1: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic = %.loc11_39.1 (constants.%.72e)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.84b) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.8ef)]
// CHECK:STDOUT:   %Inner.WithSelf.F.type: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet) [symbolic = %Inner.WithSelf.F.type (constants.%Inner.WithSelf.F.type.26a)]
// CHECK:STDOUT:   %.loc11_39.2: type = fn_type_with_self_type %Inner.WithSelf.F.type, %Inner.facet [symbolic = %.loc11_39.2 (constants.%.35d)]
// CHECK:STDOUT:   %impl.elem0.loc11_39.2: @C.as.Inner.impl.F.%.loc11_39.2 (%.35d) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_39.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_39.2: <specific function> = specific_impl_function %impl.elem0.loc11_39.2, @Inner.WithSelf.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_39.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.7d7)) -> out %return.param: @C.as.Inner.impl.F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.7d7) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_41: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_41 [symbolic = %Inner.type (constants.%Inner.type.84b)]
// CHECK:STDOUT:     %.loc11_46: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.b6b) = specific_constant @Inner.WithSelf.%assoc0.loc6_20.1, @Inner.WithSelf(constants.%T, constants.%Self.656) [symbolic = %assoc0 (constants.%assoc0.6dc)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.b6b) = name_ref F, %.loc11_46 [symbolic = %assoc0 (constants.%assoc0.6dc)]
// CHECK:STDOUT:     %impl.elem0.loc11_39.1: @C.as.Inner.impl.F.%.loc11_39.2 (%.35d) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_39.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc11_39: <bound method> = bound_method %self.ref, %impl.elem0.loc11_39.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_39.1: <specific function> = specific_impl_function %impl.elem0.loc11_39.1, @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.8ef) [symbolic = %specific_impl_fn.loc11_39.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc11_50: <bound method> = bound_method %self.ref, %specific_impl_fn.loc11_39.1
// CHECK:STDOUT:     %.loc11_24.1: ref @C.as.Inner.impl.F.%T (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %Inner.WithSelf.F.call: init @C.as.Inner.impl.F.%T (%T) to %.loc11_24.1 = call %bound_method.loc11_50(%self.ref)
// CHECK:STDOUT:     return %Inner.WithSelf.F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Inner.impl.F(%self.param: %D) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.9d0 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.9d0 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.b7f = var %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_26.2: init %C.b7f to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_3: init %C.b7f = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_3
// CHECK:STDOUT:   %.loc23_20.1: type = splice_block %C.ref [concrete = constants.%C.b7f] {
// CHECK:STDOUT:     %Outer.ref.loc23: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc23: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.e7c]
// CHECK:STDOUT:     %.loc23_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.b7f]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc23_20.2 [concrete = constants.%C.b7f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.b7f = ref_binding c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.b7f = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.e7c]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.5cc]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.5cc]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.477 = specific_constant @Inner.WithSelf.%assoc0.loc6_20.1, @Inner.WithSelf(constants.%i32, constants.%Self.656) [concrete = constants.%assoc0.7b8]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.477 = name_ref F, %.loc24_29 [concrete = constants.%assoc0.7b8]
// CHECK:STDOUT:   %impl.elem0: %.8da = impl_witness_access constants.%Inner.impl_witness.cdc, element0 [concrete = constants.%C.as.Inner.impl.F.e77]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.b7f = acquire_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %c.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%c.var)
// CHECK:STDOUT:   return %C.as.Inner.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_3.2(%self.param: ref %C.b7f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %C => constants.%C.7d7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.656
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Self.656) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.792
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.576
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.b6b
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.6dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Self.656) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%Self.as_type.abe
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.397
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.1f9
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.621
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.7d7
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.d0a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.fcc
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.1ec
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.4be
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.7d7
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.6fd
// CHECK:STDOUT:   %self.param_patt.loc11_16.2 => constants.%self.param_patt.b1b
// CHECK:STDOUT:   %self.patt.loc11_16.2 => constants.%self.patt.b71
// CHECK:STDOUT:   %.loc11_24.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc11_24 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc11_24.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc11_21.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.285) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.285
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.b47
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.350
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.b6b
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.6dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.285) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.285
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%C.7d7
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.6fd
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.b1b
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.b71
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T, constants.%Inner.facet.8ef) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.8ef
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.26a
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.3e0
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.b6b
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.6dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T, constants.%Inner.facet.8ef) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.84b
// CHECK:STDOUT:   %Self => constants.%Inner.facet.8ef
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%C.7d7
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.6fd
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.b1b
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.b71
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %C => constants.%C.b7f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.8b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Self.656) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %Self => constants.%Self.656
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.62d
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.389
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.477
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.45b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.45b
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.576
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.adb
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.477
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.45b) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.45b
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.d8d
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.f16
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.6c0
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.795
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.e1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.b7f
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.cdc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.11e
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.237
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.e77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.39b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.39b
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.a78
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.e60
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.477
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.7b8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.b7f
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.9d0
// CHECK:STDOUT:   %self.param_patt.loc11_16.2 => constants.%self.param_patt.edf
// CHECK:STDOUT:   %self.patt.loc11_16.2 => constants.%self.patt.14b
// CHECK:STDOUT:   %.loc11_24.2 => constants.%.795
// CHECK:STDOUT:   %pattern_type.loc11_24 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %return.param_patt.loc11_24.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc11_21.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_21 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %require_complete.loc11_46 => constants.%complete_type.11e
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.477
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.7b8
// CHECK:STDOUT:   %.loc11_39.1 => constants.%.fff
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.cdc
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.39b
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.a78
// CHECK:STDOUT:   %.loc11_39.2 => constants.%.8da
// CHECK:STDOUT:   %impl.elem0.loc11_39.2 => constants.%C.as.Inner.impl.F.e77
// CHECK:STDOUT:   %specific_impl_fn.loc11_39.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.39b) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.39b
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%C.b7f
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.9d0
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.edf
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.14b
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.795
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.e1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
