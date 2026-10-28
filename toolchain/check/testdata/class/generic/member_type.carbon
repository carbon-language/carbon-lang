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

class Outer(T: Core.Copy) {
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

class Outer(T: type) {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = parameterized_class_name_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.2eb: type = class_type @Outer, @Outer(%T.f84) [symbolic]
// CHECK:STDOUT:   %Inner.056: type = class_type @Inner, @Inner(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type.c1c: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Inner.elem.649: type = unbound_element_type %Inner.056, %T.as_type.c1c [symbolic]
// CHECK:STDOUT:   %struct_type.n.797: type = struct_type {.n: %T.as_type.c1c} [symbolic]
// CHECK:STDOUT:   %complete_type.22c: <witness> = complete_type_witness %struct_type.n.797 [symbolic]
// CHECK:STDOUT:   %n.param_patt.18f: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %n.patt.364: %pattern_type.2c66b4.2 = wrapper_binding_pattern n, %n.param_patt.18f [symbolic]
// CHECK:STDOUT:   %.22d: Core.Form = init_form %Inner.056 [symbolic]
// CHECK:STDOUT:   %pattern_type.ac1: type = pattern_type %Inner.056 [symbolic]
// CHECK:STDOUT:   %return.param_patt.482: %pattern_type.ac1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f04: %pattern_type.ac1 = return_slot_pattern %return.param_patt.482, %Inner.056 [symbolic]
// CHECK:STDOUT:   %Outer.F.type.5bf: type = fn_type @Outer.F, @Outer(%T.f84) [symbolic]
// CHECK:STDOUT:   %Outer.F.daa: %Outer.F.type.5bf = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.8cc: <witness> = require_complete_type %Inner.056 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d2f9c3.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.8bf: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d2f9c3.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bb5: %.8bf = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ac: <specific function> = specific_impl_function %impl.elem0.bb5, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Outer.5fe: type = class_type @Outer, @Outer(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %Inner.ea8: type = class_type @Inner, @Inner(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %Outer.F.type.777: type = fn_type @Outer.F, @Outer(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %Outer.F.01b: %Outer.F.type.777 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.1de: type = unbound_element_type %Inner.ea8, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.4d4: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.31e: <witness> = complete_type_witness %struct_type.n.4d4 [concrete]
// CHECK:STDOUT:   %pattern_type.fc8: type = pattern_type %Inner.ea8 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.fc8 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.fc8 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0b3: <witness> = impl_witness imports.%Copy.impl_witness_table.ee5 [concrete]
// CHECK:STDOUT:   %Copy.facet.d66: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.0b3) [concrete]
// CHECK:STDOUT:   %n.param_patt.1aa: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt.00d: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt.1aa [concrete]
// CHECK:STDOUT:   %.22c: Core.Form = init_form %Inner.ea8 [concrete]
// CHECK:STDOUT:   %return.param_patt.120: %pattern_type.fc8 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.815: %pattern_type.fc8 = return_slot_pattern %return.param_patt.120, %Inner.ea8 [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.01b, @Outer.F(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc13_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d1e: %SubobjectDestroy.type = facet_value %Inner.ea8, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.57f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d1e) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Inner.ea8, (%DestroyImpl.impl_witness.57f, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bbf = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.ee5 = impl_witness_table (%Core.import_ref.bbf), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
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
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.f84)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc12 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12: Core.Form = init_form %i32.loc12 [concrete = constants.%.0b3]
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
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_14.1) [symbolic = %Inner (constants.%Inner.056)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_14.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.5bf)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.5bf) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.daa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.056)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.5bf) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.daa)] {
// CHECK:STDOUT:       %n.param_patt.loc9_9.1: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %n.param_patt.loc9_9.2 (constants.%n.param_patt.18f)]
// CHECK:STDOUT:       %n.patt.loc9_9.1: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = wrapper_binding_pattern n, %n.param_patt.loc9_9.1 [symbolic = %n.patt.loc9_9.2 (constants.%n.patt.364)]
// CHECK:STDOUT:       %return.param_patt.loc9_17.1: @Outer.F.%pattern_type.loc9_17 (%pattern_type.ac1) = out_param_pattern [symbolic = %return.param_patt.loc9_17.2 (constants.%return.param_patt.482)]
// CHECK:STDOUT:       %return.patt.loc9_14.1: @Outer.F.%pattern_type.loc9_17 (%pattern_type.ac1) = return_slot_pattern %return.param_patt.loc9_17.1, %Inner.ref [symbolic = %return.patt.loc9_14.2 (constants.%return.patt.f04)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.f84) [symbolic = %Inner (constants.%Inner.056)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17.2 [symbolic = %Inner (constants.%Inner.056)]
// CHECK:STDOUT:       %.loc9_17.3: Core.Form = init_form %Inner.ref [symbolic = %.loc9_17.1 (constants.%.22d)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.3 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type.c1c)] {
// CHECK:STDOUT:         %.loc9_11.2: %Copy.type = specific_constant @Outer.%T.loc4_14.2, @Outer(constants.%T.f84) [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, %.loc9_11.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.as_type.loc9_11.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:         %.loc9_11.3: type = converted %T.ref, %T.as_type.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c) = wrapper_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.056) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.056) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.2eb
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc6_12.2: type = facet_access_type %T [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_12.2 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.056)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.as_type.loc6_12.2 [symbolic = %Inner.elem (constants.%Inner.elem.649)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.as_type.loc6_12.2 (%T.as_type.c1c)} [symbolic = %struct_type.n (constants.%struct_type.n.797)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.22c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %field_decl: @Inner.%Inner.elem (%Inner.elem.649) = field_decl n, element0, %.loc6_12.2 in [concrete] {
// CHECK:STDOUT:       %.loc6_12.1: %Copy.type = specific_constant @Outer.%T.loc4_14.2, @Outer(constants.%T.f84) [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, %.loc6_12.1 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc6_12.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:       %.loc6_12.2: type = converted %T.ref, %T.as_type.loc6_12.1 [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.797 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.22c)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.056
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc9_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type.c1c)]
// CHECK:STDOUT:   %pattern_type.loc9_9: type = pattern_type %T.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_9 (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %n.param_patt.loc9_9.2: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %n.param_patt.loc9_9.2 (constants.%n.param_patt.18f)]
// CHECK:STDOUT:   %n.patt.loc9_9.2: @Outer.F.%pattern_type.loc9_9 (%pattern_type.2c66b4.2) = wrapper_binding_pattern n, %n.param_patt.loc9_9.2 [symbolic = %n.patt.loc9_9.2 (constants.%n.patt.364)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.056)]
// CHECK:STDOUT:   %.loc9_17.1: Core.Form = init_form %Inner [symbolic = %.loc9_17.1 (constants.%.22d)]
// CHECK:STDOUT:   %pattern_type.loc9_17: type = pattern_type %Inner [symbolic = %pattern_type.loc9_17 (constants.%pattern_type.ac1)]
// CHECK:STDOUT:   %return.param_patt.loc9_17.2: @Outer.F.%pattern_type.loc9_17 (%pattern_type.ac1) = out_param_pattern [symbolic = %return.param_patt.loc9_17.2 (constants.%return.param_patt.482)]
// CHECK:STDOUT:   %return.patt.loc9_14.2: @Outer.F.%pattern_type.loc9_17 (%pattern_type.ac1) = return_slot_pattern %return.param_patt.loc9_17.2, %Inner [symbolic = %return.patt.loc9_14.2 (constants.%return.patt.f04)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.as_type.loc9_11.1 [symbolic = %require_complete.loc9_9 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.8cc)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c)} [symbolic = %struct_type.n (constants.%struct_type.n.797)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.d2f9c3.2)]
// CHECK:STDOUT:   %.loc9_38.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc9_38.3 (constants.%.8bf)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38.3 (%.8bf) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c)) -> out %return.param: @Outer.F.%Inner (%Inner.056) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.797) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38.3 (%.8bf) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.bb5)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %.loc9_38.1: %Copy.type = converted constants.%T.as_type.c1c, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc9_38.2: %Copy.type = converted constants.%T.as_type.c1c, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.2ac)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c) = class_element_access %return.param, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c) to %.loc9_39.2 = call %bound_method.loc9_38.2(%n.ref)
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type.c1c) to %.loc9_39.2 = in_place_init %Copy.WithSelf.Op.call
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.056) to %return.param = class_init (%.loc9_39.3)
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.056) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %Inner.ea8 = var_storage %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc13_35: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value %i32.loc13_35, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.95d) [concrete = constants.%Outer.5fe]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.777 = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.95d) [concrete = constants.%Outer.F.01b]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.777 = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.01b]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43.1: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.0b3) [concrete = constants.%Copy.facet.d66]
// CHECK:STDOUT:   %.loc13_43.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.1 [concrete = constants.%Copy.facet.d66]
// CHECK:STDOUT:   %Copy.facet.loc13_43.2: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.0b3) [concrete = constants.%Copy.facet.d66]
// CHECK:STDOUT:   %.loc13_43.2: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.2 [concrete = constants.%Copy.facet.d66]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.95d) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3.1: ref %Inner.ea8 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0.loc13: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.ea8 to %.loc13_3.1 = call %Outer.F.specific_fn(%.loc13_42.2)
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.ea8] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc13_16: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value %i32.loc13_16, (constants.%Copy.impl_witness.8af) [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.95d]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.95d) [concrete = constants.%Outer.5fe]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.95d) [concrete = constants.%Inner.ea8]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.ea8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.ea8 = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.fc8 = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.fc8 = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: ref %Inner.ea8 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.1de = name_ref n, @Inner.%field_decl [concrete = @Inner.%field_decl]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = acquire_value %.loc14_11.1
// CHECK:STDOUT:   %impl.elem0.loc14: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_fn.loc14
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%Inner.ea8, (constants.%DestroyImpl.impl_witness.57f, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%Inner.ea8, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%Inner.ea8, (constants.%DestroyImpl.impl_witness.57f, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%Inner.ea8, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%c.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.3(%self.param: ref %struct_type.n.4d4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc13_3.4(%self.param: ref %Inner.ea8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.f84
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.056
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.5bf
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.daa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.f84) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc6_12.2 => constants.%T.as_type.c1c
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %Inner => constants.%Inner.056
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.649
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.797
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.22c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%T.as_type.c1c
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %n.param_patt.loc9_9.2 => constants.%n.param_patt.18f
// CHECK:STDOUT:   %n.patt.loc9_9.2 => constants.%n.patt.364
// CHECK:STDOUT:   %Inner => constants.%Inner.056
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.22d
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.ac1
// CHECK:STDOUT:   %return.param_patt.loc9_17.2 => constants.%return.param_patt.482
// CHECK:STDOUT:   %return.patt.loc9_14.2 => constants.%return.patt.f04
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.95d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.ea8
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.777
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.01b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.95d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.95d
// CHECK:STDOUT:   %T.as_type.loc6_12.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.ea8
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.1de
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.4d4
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.31e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.95d) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.95d
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_9 => constants.%pattern_type.d94
// CHECK:STDOUT:   %n.param_patt.loc9_9.2 => constants.%n.param_patt.1aa
// CHECK:STDOUT:   %n.patt.loc9_9.2 => constants.%n.patt.00d
// CHECK:STDOUT:   %Inner => constants.%Inner.ea8
// CHECK:STDOUT:   %.loc9_17.1 => constants.%.22c
// CHECK:STDOUT:   %pattern_type.loc9_17 => constants.%pattern_type.fc8
// CHECK:STDOUT:   %return.param_patt.loc9_17.2 => constants.%return.param_patt.120
// CHECK:STDOUT:   %return.patt.loc9_14.2 => constants.%return.patt.815
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.31e
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.4d4
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.f76
// CHECK:STDOUT:   %.loc9_38.3 => constants.%.2b9
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.8af
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.cae
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a587c.1: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = parameterized_class_name_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.651: type = class_type @Outer, @Outer(%T.67d) [symbolic]
// CHECK:STDOUT:   %Inner.type.ebc: type = facet_type <@Inner, @Inner(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.6a1: %Inner.type.ebc = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.ef4: type = facet_access_type %Self.6a1 [symbolic]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %Self.as_type.ef4 [symbolic]
// CHECK:STDOUT:   %self.param_patt.ac7: %pattern_type.44a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d58: %pattern_type.44a = wrapper_binding_pattern self, %self.param_patt.ac7 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T.67d [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.d13: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T.67d, %Self.6a1) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.98c: %Inner.WithSelf.F.type.d13 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.feb: type = assoc_entity_type @Inner, @Inner(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.3e8: %Inner.assoc_type.feb = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %C.128: type = class_type @C, @C(%T.67d) [symbolic]
// CHECK:STDOUT:   %.1cb: <witness> = impl_self_witness %C.128, @Inner, @Inner(%T.67d) [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.19c: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.074: <witness> = require_complete_type %Inner.type.ebc [symbolic]
// CHECK:STDOUT:   %pattern_type.dc7: type = pattern_type %C.128 [symbolic]
// CHECK:STDOUT:   %self.param_patt.385: %pattern_type.dc7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.f31: %pattern_type.dc7 = wrapper_binding_pattern self, %self.param_patt.385 [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.70c: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.7b0: %C.as.Inner.impl.F.type.70c = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.04b: %Inner.type.ebc = facet_value %C.128, (%Inner.impl_witness.19c) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.dc2: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T.67d, %Inner.facet.04b) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.600: %Inner.WithSelf.F.type.dc2 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.e45: <witness> = require_complete_type %C.128 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.128, @Inner, @Inner(%T.67d) [symbolic]
// CHECK:STDOUT:   %.f55: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Inner.facet.d73: %Inner.type.ebc = facet_value %C.128, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.001: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T.67d, %Inner.facet.d73) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.5d3: %Inner.WithSelf.F.type.001 = struct_value () [symbolic]
// CHECK:STDOUT:   %.bab: type = fn_type_with_self_type %Inner.WithSelf.F.type.001, %Inner.facet.d73 [symbolic]
// CHECK:STDOUT:   %impl.elem0.eb0: %.bab = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.eef: <specific function> = specific_impl_function %impl.elem0.eb0, @Inner.WithSelf.F(%T.67d, %Inner.facet.d73) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.7d3: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.b91: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.785: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %.79a: <witness> = impl_self_witness %D, @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.639: <witness> = impl_witness @D.as.Inner.impl.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.77b: %Inner.type.b91 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.233: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Self.6a1) [symbolic]
// CHECK:STDOUT:   %Inner.WithSelf.F.4f0: %Inner.WithSelf.F.type.233 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.0c1: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.1ea: %Inner.assoc_type.0c1 = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %self.param_patt.f16: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.752: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt.f16 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F.type: type = fn_type @D.as.Inner.impl.F [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F: %D.as.Inner.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.faa: %Inner.type.b91 = facet_value %D, (%Inner.impl_witness.639) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.655: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.faa) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.9d2: %Inner.WithSelf.F.type.655 = struct_value () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.e4b: type = pattern_type %C.785 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.e4b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.e4b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C.785 = struct_value () [concrete]
// CHECK:STDOUT:   %.1b6: <witness> = impl_self_witness %C.785, @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.728: <witness> = impl_witness @C.as.Inner.impl.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %complete_type.469: <witness> = complete_type_witness %Inner.type.b91 [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.460: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.e15: %C.as.Inner.impl.F.type.460 = struct_value () [concrete]
// CHECK:STDOUT:   %.ee5: require_specific_def_type = require_specific_def @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %Inner.facet.e46: %Inner.type.b91 = facet_value %C.785, (%Inner.impl_witness.728) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.type.a68: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%i32, %Inner.facet.e46) [concrete]
// CHECK:STDOUT:   %Inner.WithSelf.F.6d2: %Inner.WithSelf.F.type.a68 = struct_value () [concrete]
// CHECK:STDOUT:   %.0c4: type = fn_type_with_self_type %Inner.WithSelf.F.type.a68, %Inner.facet.e46 [concrete]
// CHECK:STDOUT:   %self.param_patt.17c: %pattern_type.e4b = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.894: %pattern_type.e4b = wrapper_binding_pattern self, %self.param_patt.17c [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.e15, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.905: %SubobjectDestroy.type = facet_value %C.785, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.ad6: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.905) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C.785, (%DestroyImpl.impl_witness.ad6, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc22 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc22: Core.Form = init_form %i32.loc22 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:   %Self.loc5_19.2: @Inner.%Inner.type (%Inner.type.ebc) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.6a1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.ebc) = symbolic_binding Self, 1 [symbolic = %Self.loc5_19.2 (constants.%Self.6a1)]
// CHECK:STDOUT:     %Inner.WithSelf.decl = interface_with_self_decl @Inner [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %Inner.WithSelf.F.decl: @Inner.WithSelf.%Inner.WithSelf.F.type (%Inner.WithSelf.F.type.d13) = fn_decl @Inner.WithSelf.F [symbolic = @Inner.WithSelf.%Inner.WithSelf.F (constants.%Inner.WithSelf.F.98c)] {
// CHECK:STDOUT:       %self.param_patt.loc6_10.1: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.44a) = value_param_pattern [symbolic = %self.param_patt.loc6_10.2 (constants.%self.param_patt.ac7)]
// CHECK:STDOUT:       %self.patt.loc6_10.1: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.44a) = wrapper_binding_pattern self, %self.param_patt.loc6_10.1 [symbolic = %self.patt.loc6_10.2 (constants.%self.patt.d58)]
// CHECK:STDOUT:       %return.param_patt.loc6_19.1: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_19.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc6_16.1: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_19.1, %T.ref [symbolic = %return.patt.loc6_16.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc6_19.2: type = specific_constant @Outer.%T.loc4_14.2, @Outer(constants.%T.67d) [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %.loc6_19.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc6_19.3: Core.Form = init_form %T.ref [symbolic = %.loc6_19.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @Inner.WithSelf.F.%Self.as_type.loc6_10.1 (%Self.as_type.ef4) = value_param call_param0
// CHECK:STDOUT:       %.loc6_10.1: type = splice_block %.loc6_10.3 [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.ef4)] {
// CHECK:STDOUT:         %.loc6_10.2: @Inner.WithSelf.F.%Inner.type (%Inner.type.ebc) = specific_constant @Inner.%Self.loc5_19.1, @Inner(constants.%T.67d) [symbolic = %Self (constants.%Self.6a1)]
// CHECK:STDOUT:         %Self.ref: @Inner.WithSelf.F.%Inner.type (%Inner.type.ebc) = name_ref Self, %.loc6_10.2 [symbolic = %Self (constants.%Self.6a1)]
// CHECK:STDOUT:         %Self.as_type.loc6_10.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.ef4)]
// CHECK:STDOUT:         %.loc6_10.3: type = converted %Self.ref, %Self.as_type.loc6_10.2 [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.ef4)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.WithSelf.F.%Self.as_type.loc6_10.1 (%Self.as_type.ef4) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Inner.WithSelf.F.%T (%T.67d) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Inner.WithSelf.F.%T (%T.67d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_20.1: @Inner.WithSelf.%Inner.assoc_type (%Inner.assoc_type.feb) = assoc_entity element0, %Inner.WithSelf.F.decl [symbolic = %assoc0.loc6_20.2 (constants.%assoc0.3e8)]
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.128)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:   %.loc10_19: <witness> = impl_self_witness %C, @Inner, @Inner(%T) [symbolic = %.loc10_19 (constants.%.1cb)]
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.19c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.074)]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.70c)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.70c) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.7b0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.70c) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.7b0)] {
// CHECK:STDOUT:       %self.param_patt.loc11_16.1: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.dc7) = value_param_pattern [symbolic = %self.param_patt.loc11_16.2 (constants.%self.param_patt.385)]
// CHECK:STDOUT:       %self.patt.loc11_16.1: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.dc7) = wrapper_binding_pattern self, %self.param_patt.loc11_16.1 [symbolic = %self.patt.loc11_16.2 (constants.%self.patt.f31)]
// CHECK:STDOUT:       %return.param_patt.loc11_24.1: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc11_24.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc11_21.1: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc11_24.1, %T.ref [symbolic = %return.patt.loc11_21.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc11_24.3: type = specific_constant @Outer.%T.loc4_14.2, @Outer(constants.%T.67d) [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %.loc11_24.3 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc11_24.4: Core.Form = init_form %T.ref [symbolic = %.loc11_24.2 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @C.as.Inner.impl.F.%C (%C.128) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.128)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T.67d) [symbolic = %C (constants.%C.128)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.128)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Inner.impl.F.%C (%C.128) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @C.as.Inner.impl.F.%T (%T.67d) = out_param call_param1
// CHECK:STDOUT:       %return: ref @C.as.Inner.impl.F.%T (%T.67d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness.loc10_19.1: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T.67d) [symbolic = %Inner.impl_witness.loc10_19.2 (constants.%Inner.impl_witness.19c)]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.752]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %self.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %self: %D = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (%D.as.Inner.impl.F.decl), @D.as.Inner.impl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.639]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT:   .F = %D.as.Inner.impl.F.decl
// CHECK:STDOUT:   extend %Inner.ref
// CHECK:STDOUT:   witness = %Inner.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.9a587c.1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_14.1)> [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_14.1) [symbolic = %C (constants.%C.128)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.ebc)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.128)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.651
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
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.128 [symbolic = %C (constants.%C.128)]
// CHECK:STDOUT:       %.loc10_13: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.67d) [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10_13 [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc10: <witness> = impl_self_witness @C.as.Inner.impl.%Self.ref, @Inner, @Inner(constants.%T.67d) [symbolic = @C.as.Inner.impl.%.loc10_19 (constants.%.1cb)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.128
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
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.7d3]
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.b91]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.b91]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc17: <witness> = impl_self_witness @D.as.Inner.impl.%Self.ref, @Inner, @Inner(constants.%i32) [concrete = constants.%.79a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.WithSelf.F(@Outer.%T.loc4_14.2: type, @Inner.%Self.loc5_19.1: @Inner.%Inner.type (%Inner.type.ebc)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:   %Self: @Inner.WithSelf.F.%Inner.type (%Inner.type.ebc) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.6a1)]
// CHECK:STDOUT:   %Self.as_type.loc6_10.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_10.1 (constants.%Self.as_type.ef4)]
// CHECK:STDOUT:   %pattern_type.loc6_10: type = pattern_type %Self.as_type.loc6_10.1 [symbolic = %pattern_type.loc6_10 (constants.%pattern_type.44a)]
// CHECK:STDOUT:   %self.param_patt.loc6_10.2: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.44a) = value_param_pattern [symbolic = %self.param_patt.loc6_10.2 (constants.%self.param_patt.ac7)]
// CHECK:STDOUT:   %self.patt.loc6_10.2: @Inner.WithSelf.F.%pattern_type.loc6_10 (%pattern_type.44a) = wrapper_binding_pattern self, %self.param_patt.loc6_10.2 [symbolic = %self.patt.loc6_10.2 (constants.%self.patt.d58)]
// CHECK:STDOUT:   %.loc6_19.1: Core.Form = init_form %T [symbolic = %.loc6_19.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_19: type = pattern_type %T [symbolic = %pattern_type.loc6_19 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc6_19.2: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc6_19.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc6_16.2: @Inner.WithSelf.F.%pattern_type.loc6_19 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc6_19.2, %T [symbolic = %return.patt.loc6_16.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.WithSelf.F.%Self.as_type.loc6_10.1 (%Self.as_type.ef4)) -> out %return.param: @Inner.WithSelf.F.%T (%T.67d);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.128)]
// CHECK:STDOUT:   %pattern_type.loc11_16: type = pattern_type %C [symbolic = %pattern_type.loc11_16 (constants.%pattern_type.dc7)]
// CHECK:STDOUT:   %self.param_patt.loc11_16.2: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.dc7) = value_param_pattern [symbolic = %self.param_patt.loc11_16.2 (constants.%self.param_patt.385)]
// CHECK:STDOUT:   %self.patt.loc11_16.2: @C.as.Inner.impl.F.%pattern_type.loc11_16 (%pattern_type.dc7) = wrapper_binding_pattern self, %self.param_patt.loc11_16.2 [symbolic = %self.patt.loc11_16.2 (constants.%self.patt.f31)]
// CHECK:STDOUT:   %.loc11_24.2: Core.Form = init_form %T [symbolic = %.loc11_24.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc11_24: type = pattern_type %T [symbolic = %pattern_type.loc11_24 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc11_24.2: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc11_24.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc11_21.2: @C.as.Inner.impl.F.%pattern_type.loc11_24 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc11_24.2, %T [symbolic = %return.patt.loc11_21.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.e45)]
// CHECK:STDOUT:   %require_complete.loc11_21: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_21 (constants.%require_complete.944)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:   %require_complete.loc11_46: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_46 (constants.%require_complete.074)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.feb)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.feb) = assoc_entity element0, @Inner.WithSelf.%Inner.WithSelf.F.decl [symbolic = %assoc0 (constants.%assoc0.3e8)]
// CHECK:STDOUT:   %.loc11_39.1: require_specific_def_type = require_specific_def @C.as.Inner.impl(%T) [symbolic = %.loc11_39.1 (constants.%.f55)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.ebc) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.d73)]
// CHECK:STDOUT:   %Inner.WithSelf.F.type: type = fn_type @Inner.WithSelf.F, @Inner.WithSelf(%T, %Inner.facet) [symbolic = %Inner.WithSelf.F.type (constants.%Inner.WithSelf.F.type.001)]
// CHECK:STDOUT:   %.loc11_39.2: type = fn_type_with_self_type %Inner.WithSelf.F.type, %Inner.facet [symbolic = %.loc11_39.2 (constants.%.bab)]
// CHECK:STDOUT:   %impl.elem0.loc11_39.2: @C.as.Inner.impl.F.%.loc11_39.2 (%.bab) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_39.2 (constants.%impl.elem0.eb0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_39.2: <specific function> = specific_impl_function %impl.elem0.loc11_39.2, @Inner.WithSelf.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_39.2 (constants.%specific_impl_fn.eef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.128)) -> out %return.param: @C.as.Inner.impl.F.%T (%T.67d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.128) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_41: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.67d) [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_41 [symbolic = %Inner.type (constants.%Inner.type.ebc)]
// CHECK:STDOUT:     %.loc11_46: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.feb) = specific_constant @Inner.WithSelf.%assoc0.loc6_20.1, @Inner.WithSelf(constants.%T.67d, constants.%Self.6a1) [symbolic = %assoc0 (constants.%assoc0.3e8)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.feb) = name_ref F, %.loc11_46 [symbolic = %assoc0 (constants.%assoc0.3e8)]
// CHECK:STDOUT:     %impl.elem0.loc11_39.1: @C.as.Inner.impl.F.%.loc11_39.2 (%.bab) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_39.2 (constants.%impl.elem0.eb0)]
// CHECK:STDOUT:     %bound_method.loc11_39: <bound method> = bound_method %self.ref, %impl.elem0.loc11_39.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_39.1: <specific function> = specific_impl_function %impl.elem0.loc11_39.1, @Inner.WithSelf.F(constants.%T.67d, constants.%Inner.facet.d73) [symbolic = %specific_impl_fn.loc11_39.2 (constants.%specific_impl_fn.eef)]
// CHECK:STDOUT:     %bound_method.loc11_50: <bound method> = bound_method %self.ref, %specific_impl_fn.loc11_39.1
// CHECK:STDOUT:     %.loc11_24.1: ref @C.as.Inner.impl.F.%T (%T.67d) = splice_block %return.param {}
// CHECK:STDOUT:     %Inner.WithSelf.F.call: init @C.as.Inner.impl.F.%T (%T.67d) to %.loc11_24.1 = call %bound_method.loc11_50(%self.ref)
// CHECK:STDOUT:     return %Inner.WithSelf.F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Inner.impl.F(%self.param: %D) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.var: ref %C.785 = var_storage %c.var_patt
// CHECK:STDOUT:   %.loc23_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc23_26.2: init %C.785 to %c.var = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc23_3.1: init %C.785 = converted %.loc23_26.1, %.loc23_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc23_3.1
// CHECK:STDOUT:   %.loc23_20.1: type = splice_block %C.ref [concrete = constants.%C.785] {
// CHECK:STDOUT:     %Outer.ref.loc23: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %i32.loc23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc23: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.7d3]
// CHECK:STDOUT:     %.loc23_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.785]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc23_20.2 [concrete = constants.%C.785]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.785 = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.e4b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.e4b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: ref %C.785 = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32.loc24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.7d3]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.b91]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.b91]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.0c1 = specific_constant @Inner.WithSelf.%assoc0.loc6_20.1, @Inner.WithSelf(constants.%i32, constants.%Self.6a1) [concrete = constants.%assoc0.1ea]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.0c1 = name_ref F, %.loc24_29 [concrete = constants.%assoc0.1ea]
// CHECK:STDOUT:   %impl.elem0: %.0c4 = impl_witness_access constants.%Inner.impl_witness.728, element0 [concrete = constants.%C.as.Inner.impl.F.e15]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.785 = acquire_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %Destroy.facet.loc23_3.1: %Destroy.type = facet_value constants.%C.785, (constants.%DestroyImpl.impl_witness.ad6, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc23_3.2: %Destroy.type = converted constants.%C.785, %Destroy.facet.loc23_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc23_3.2: %Destroy.type = facet_value constants.%C.785, (constants.%DestroyImpl.impl_witness.ad6, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc23_3.3: %Destroy.type = converted constants.%C.785, %Destroy.facet.loc23_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%c.var)
// CHECK:STDOUT:   return %C.as.Inner.impl.F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23_3.2(%self.param: ref %C.785) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %C => constants.%C.128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.67d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.6a1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T.67d, constants.%Self.6a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %Self => constants.%Self.6a1
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.d13
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.98c
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.feb
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.3e8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T.67d, constants.%Self.6a1) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %Self => constants.%Self.6a1
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%Self.as_type.ef4
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.44a
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.ac7
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.d58
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T.67d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %C => constants.%C.128
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %.loc10_19 => constants.%.1cb
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.19c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.074
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.70c
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.7b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %C => constants.%C.128
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.dc7
// CHECK:STDOUT:   %self.param_patt.loc11_16.2 => constants.%self.param_patt.385
// CHECK:STDOUT:   %self.patt.loc11_16.2 => constants.%self.patt.f31
// CHECK:STDOUT:   %.loc11_24.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc11_24 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc11_24.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc11_21.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T.67d, constants.%Inner.facet.04b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.04b
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.dc2
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.600
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.feb
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.3e8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T.67d, constants.%Inner.facet.04b) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.04b
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%C.128
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.dc7
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.385
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.f31
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%T.67d, constants.%Inner.facet.d73) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.d73
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.001
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.5d3
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.feb
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.3e8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%T.67d, constants.%Inner.facet.d73) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.ebc
// CHECK:STDOUT:   %Self => constants.%Inner.facet.d73
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%C.128
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.dc7
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.385
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.f31
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %C => constants.%C.785
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %Self.loc5_19.2 => constants.%Self.77b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Self.6a1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %Self => constants.%Self.6a1
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.233
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.4f0
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.0c1
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.1ea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.faa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %Self => constants.%Inner.facet.faa
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.655
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.9d2
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.0c1
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.1ea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.faa) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %Self => constants.%Inner.facet.faa
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.d8d
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.f16
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.752
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.0b3
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.d94
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.6fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.785
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %.loc10_19 => constants.%.1b6
// CHECK:STDOUT:   %Inner.impl_witness.loc10_19.2 => constants.%Inner.impl_witness.728
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.469
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.460
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.e15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf(constants.%i32, constants.%Inner.facet.e46) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %Self => constants.%Inner.facet.e46
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.a68
// CHECK:STDOUT:   %Inner.WithSelf.F => constants.%Inner.WithSelf.F.6d2
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.0c1
// CHECK:STDOUT:   %assoc0.loc6_20.2 => constants.%assoc0.1ea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.785
// CHECK:STDOUT:   %pattern_type.loc11_16 => constants.%pattern_type.e4b
// CHECK:STDOUT:   %self.param_patt.loc11_16.2 => constants.%self.param_patt.17c
// CHECK:STDOUT:   %self.patt.loc11_16.2 => constants.%self.patt.894
// CHECK:STDOUT:   %.loc11_24.2 => constants.%.0b3
// CHECK:STDOUT:   %pattern_type.loc11_24 => constants.%pattern_type.d94
// CHECK:STDOUT:   %return.param_patt.loc11_24.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc11_21.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_16 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc11_21 => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %require_complete.loc11_46 => constants.%complete_type.469
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.0c1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.1ea
// CHECK:STDOUT:   %.loc11_39.1 => constants.%.ee5
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.728
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.e46
// CHECK:STDOUT:   %Inner.WithSelf.F.type => constants.%Inner.WithSelf.F.type.a68
// CHECK:STDOUT:   %.loc11_39.2 => constants.%.0c4
// CHECK:STDOUT:   %impl.elem0.loc11_39.2 => constants.%C.as.Inner.impl.F.e15
// CHECK:STDOUT:   %specific_impl_fn.loc11_39.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.WithSelf.F(constants.%i32, constants.%Inner.facet.e46) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.b91
// CHECK:STDOUT:   %Self => constants.%Inner.facet.e46
// CHECK:STDOUT:   %Self.as_type.loc6_10.1 => constants.%C.785
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.e4b
// CHECK:STDOUT:   %self.param_patt.loc6_10.2 => constants.%self.param_patt.17c
// CHECK:STDOUT:   %self.patt.loc6_10.2 => constants.%self.patt.894
// CHECK:STDOUT:   %.loc6_19.1 => constants.%.0b3
// CHECK:STDOUT:   %pattern_type.loc6_19 => constants.%pattern_type.d94
// CHECK:STDOUT:   %return.param_patt.loc6_19.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc6_16.2 => constants.%return.patt.6fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
