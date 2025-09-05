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
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.a20: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.00c: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.7ce: type = class_type @Outer, @Outer(%T.a20) [symbolic]
// CHECK:STDOUT:   %Inner.e67: type = class_type @Inner, @Inner(%T.a20) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.a20 [symbolic]
// CHECK:STDOUT:   %require_complete.909: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.elem.d4d: type = unbound_element_type %Inner.e67, %T.as_type [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.30c: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%T.a20) [symbolic]
// CHECK:STDOUT:   %ptr.a76: type = ptr_type %Inner.e67 [symbolic]
// CHECK:STDOUT:   %pattern_type.669: type = pattern_type %ptr.a76 [symbolic]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type.3a6: type = fn_type @Inner.as.Destroy.impl.Op, @Inner.as.Destroy.impl(%T.a20) [symbolic]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.c24: %Inner.as.Destroy.impl.Op.type.3a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.n.744: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.09e: <witness> = complete_type_witness %struct_type.n.744 [symbolic]
// CHECK:STDOUT:   %pattern_type.47fbd7.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.45d: type = pattern_type %Inner.e67 [symbolic]
// CHECK:STDOUT:   %Outer.F.type.0d7: type = fn_type @Outer.F, @Outer(%T.a20) [symbolic]
// CHECK:STDOUT:   %Outer.F.f56: %Outer.F.type.0d7 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.c9f: <witness> = impl_witness @Outer.%Destroy.impl_witness_table, @Outer.as.Destroy.impl(%T.a20) [symbolic]
// CHECK:STDOUT:   %ptr.1aa: type = ptr_type %Outer.7ce [symbolic]
// CHECK:STDOUT:   %pattern_type.c68: type = pattern_type %ptr.1aa [symbolic]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op, @Outer.as.Destroy.impl(%T.a20) [symbolic]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: %Outer.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.36b: <witness> = require_complete_type %Inner.e67 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.986: <witness> = lookup_impl_witness %T.a20, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.f95: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.986) [symbolic]
// CHECK:STDOUT:   %.fdd: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.f95 [symbolic]
// CHECK:STDOUT:   %impl.elem0.26e: %.fdd = impl_witness_access %Copy.lookup_impl_witness.986, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.930: <specific function> = specific_impl_function %impl.elem0.26e, @Copy.Op(%Copy.facet.f95) [symbolic]
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
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.e1f: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.0ec: %Int.as.Copy.impl.Op.type.e1f = struct_value () [symbolic]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.dcf: <witness> = impl_witness imports.%Copy.impl_witness_table.41f, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.db7: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.d19: %Int.as.Copy.impl.Op.type.db7 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.715: %Copy.type = facet_value %i32, (%Copy.impl_witness.dcf) [concrete]
// CHECK:STDOUT:   %Outer.205: type = class_type @Outer, @Outer(%Copy.facet.715) [concrete]
// CHECK:STDOUT:   %Inner.576: type = class_type @Inner, @Inner(%Copy.facet.715) [concrete]
// CHECK:STDOUT:   %Outer.F.type.d27: type = fn_type @Outer.F, @Outer(%Copy.facet.715) [concrete]
// CHECK:STDOUT:   %Outer.F.833: %Outer.F.type.d27 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.ba1: type = unbound_element_type %Inner.576, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.033: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n.033 [concrete]
// CHECK:STDOUT:   %pattern_type.e38: type = pattern_type %Inner.576 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.36a: <witness> = impl_witness imports.%Copy.impl_witness_table.929 [concrete]
// CHECK:STDOUT:   %Copy.facet.912: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.36a) [concrete]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %Outer.F.833, @Outer.F(%Copy.facet.715) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.cad: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%Copy.facet.715) [concrete]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type.7a4: type = fn_type @Inner.as.Destroy.impl.Op, @Inner.as.Destroy.impl(%Copy.facet.715) [concrete]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.9b6: %Inner.as.Destroy.impl.Op.type.7a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c3e: type = ptr_type %Inner.576 [concrete]
// CHECK:STDOUT:   %pattern_type.55d: type = pattern_type %ptr.c3e [concrete]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Inner.as.Destroy.impl.Op.9b6, @Inner.as.Destroy.impl.Op(%Copy.facet.715) [concrete]
// CHECK:STDOUT:   %.8ee: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.715 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.d19, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.ae0: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.e1f) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.0ec)]
// CHECK:STDOUT:   %Copy.impl_witness_table.41f = impl_witness_table (%Core.import_ref.ae0), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.58b = import_ref Core//prelude/parts/copy, loc28_31, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.929 = impl_witness_table (%Core.import_ref.58b), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.00c = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T.a20)]
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
// CHECK:STDOUT: generic impl @Inner.as.Destroy.impl(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.30c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type: type = fn_type @Inner.as.Destroy.impl.Op, @Inner.as.Destroy.impl(%T) [symbolic = %Inner.as.Destroy.impl.Op.type (constants.%Inner.as.Destroy.impl.Op.type.3a6)]
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op: @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.type (%Inner.as.Destroy.impl.Op.type.3a6) = struct_value () [symbolic = %Inner.as.Destroy.impl.Op (constants.%Inner.as.Destroy.impl.Op.c24)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Inner.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Inner.as.Destroy.impl.Op.decl: @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.type (%Inner.as.Destroy.impl.Op.type.3a6) = fn_decl @Inner.as.Destroy.impl.Op [symbolic = @Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op (constants.%Inner.as.Destroy.impl.Op.c24)] {
// CHECK:STDOUT:       %self.patt: @Inner.as.Destroy.impl.Op.%pattern_type (%pattern_type.669) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Inner.as.Destroy.impl.Op.%pattern_type (%pattern_type.669) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc5_15.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Inner.as.Destroy.impl.Op.%ptr (%ptr.a76) = value_param call_param0
// CHECK:STDOUT:       %.loc5_15.2: type = splice_block %Self.ref [symbolic = %Inner (constants.%Inner.e67)] {
// CHECK:STDOUT:         %.loc5_15.3: type = specific_constant constants.%Inner.e67, @Inner(constants.%T.a20) [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_15.3 [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.as.Destroy.impl.Op.%ptr (%ptr.a76) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Inner.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Inner.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Outer.as.Destroy.impl(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%T) [symbolic = %Outer (constants.%Outer.7ce)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Outer.%Destroy.impl_witness_table, @Outer.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.c9f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op, @Outer.as.Destroy.impl(%T) [symbolic = %Outer.as.Destroy.impl.Op.type (constants.%Outer.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.type (%Outer.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Outer.as.Destroy.impl.Op (constants.%Outer.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Outer.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Outer.as.Destroy.impl.Op.decl: @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.type (%Outer.as.Destroy.impl.Op.type) = fn_decl @Outer.as.Destroy.impl.Op [symbolic = @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op (constants.%Outer.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Outer.as.Destroy.impl.Op.%pattern_type (%pattern_type.c68) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Outer.as.Destroy.impl.Op.%pattern_type (%pattern_type.c68) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_28.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Outer.as.Destroy.impl.Op.%ptr (%ptr.1aa) = value_param call_param0
// CHECK:STDOUT:       %.loc4_28.2: type = splice_block %Self.ref [symbolic = %Outer (constants.%Outer.7ce)] {
// CHECK:STDOUT:         %.loc4_28.3: type = specific_constant constants.%Outer.7ce, @Outer(constants.%T.a20) [symbolic = %Outer (constants.%Outer.7ce)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_28.3 [symbolic = %Outer (constants.%Outer.7ce)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Outer.as.Destroy.impl.Op.%ptr (%ptr.1aa) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Outer.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Outer.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc4_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T.a20)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_13.1) [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:   %Outer.F.type: type = fn_type @Outer.F, @Outer(%T.loc4_13.1) [symbolic = %Outer.F.type (constants.%Outer.F.type.0d7)]
// CHECK:STDOUT:   %Outer.F: @Outer.%Outer.F.type (%Outer.F.type.0d7) = struct_value () [symbolic = %Outer.F (constants.%Outer.F.f56)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.e67)] {} {}
// CHECK:STDOUT:     %Outer.F.decl: @Outer.%Outer.F.type (%Outer.F.type.0d7) = fn_decl @Outer.F [symbolic = @Outer.%Outer.F (constants.%Outer.F.f56)] {
// CHECK:STDOUT:       %n.patt: @Outer.F.%pattern_type.loc9_8 (%pattern_type.47fbd7.1) = binding_pattern n [concrete]
// CHECK:STDOUT:       %n.param_patt: @Outer.F.%pattern_type.loc9_8 (%pattern_type.47fbd7.1) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Outer.F.%pattern_type.loc9_14 (%pattern_type.45d) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Outer.F.%pattern_type.loc9_14 (%pattern_type.45d) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T.a20) [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17 [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:       %n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.1: type = splice_block %.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:         %T.as_type.loc9_11.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc9_11.2: type = converted %T.ref, %T.as_type.loc9_11.2 [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = bind_name n, %n.param
// CHECK:STDOUT:       %return.param: ref @Outer.F.%Inner (%Inner.e67) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Outer.F.%Inner (%Inner.e67) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Outer.7ce [symbolic = @Outer.as.Destroy.impl.%Outer (constants.%Outer.7ce)]
// CHECK:STDOUT:     impl_decl @Outer.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.decl), @Outer.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Outer.as.Destroy.impl(constants.%T.a20) [symbolic = @Outer.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.c9f)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.7ce
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Outer.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:   %T.as_type.loc6_12.2: type = facet_access_type %T [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_12.2 [symbolic = %require_complete (constants.%require_complete.909)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T.as_type.loc6_12.2 [symbolic = %Inner.elem (constants.%Inner.elem.d4d)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Inner.%T.as_type.loc6_12.2 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.744)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.09e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:     %T.as_type.loc6_12.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6_12: type = converted %T.ref, %T.as_type.loc6_12.1 [symbolic = %T.as_type.loc6_12.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc6_10: @Inner.%Inner.elem (%Inner.elem.d4d) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Inner.e67 [symbolic = @Inner.as.Destroy.impl.%Inner (constants.%Inner.e67)]
// CHECK:STDOUT:     impl_decl @Inner.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Inner.as.Destroy.impl.%Inner.as.Destroy.impl.Op.decl), @Inner.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Inner.as.Destroy.impl(constants.%T.a20) [symbolic = @Inner.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.30c)]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness constants.%struct_type.n.744 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.09e)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.e67
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6_10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Inner [symbolic = %ptr (constants.%ptr.a76)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.669)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.as.Destroy.impl.Op.%ptr (%ptr.a76)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.F(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:   %T.as_type.loc9_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc9_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9_8: type = pattern_type %T.as_type.loc9_11.1 [symbolic = %pattern_type.loc9_8 (constants.%pattern_type.47fbd7.1)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.e67)]
// CHECK:STDOUT:   %pattern_type.loc9_14: type = pattern_type %Inner [symbolic = %pattern_type.loc9_14 (constants.%pattern_type.45d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.36b)]
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T.as_type.loc9_11.1 [symbolic = %require_complete.loc9_9 (constants.%require_complete.909)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n.744)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.986)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc9_11.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.f95)]
// CHECK:STDOUT:   %.loc9_38.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc9_38.2 (constants.%.fdd)]
// CHECK:STDOUT:   %impl.elem0.loc9_38.2: @Outer.F.%.loc9_38.2 (%.fdd) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.26e)]
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2: <specific function> = specific_impl_function %impl.elem0.loc9_38.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.930)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type)) -> %return.param: @Outer.F.%Inner (%Inner.e67) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @Outer.F.%struct_type.n (%struct_type.n.744) = struct_literal (%n.ref)
// CHECK:STDOUT:     %impl.elem0.loc9_38.1: @Outer.F.%.loc9_38.2 (%.fdd) = impl_witness_access constants.%Copy.lookup_impl_witness.986, element0 [symbolic = %impl.elem0.loc9_38.2 (constants.%impl.elem0.26e)]
// CHECK:STDOUT:     %bound_method.loc9_38.1: <bound method> = bound_method %n.ref, %impl.elem0.loc9_38.1
// CHECK:STDOUT:     %specific_impl_fn.loc9_38.1: <specific function> = specific_impl_function %impl.elem0.loc9_38.1, @Copy.Op(constants.%Copy.facet.f95) [symbolic = %specific_impl_fn.loc9_38.2 (constants.%specific_impl_fn.930)]
// CHECK:STDOUT:     %bound_method.loc9_38.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc9_38.1
// CHECK:STDOUT:     %.loc9_38.1: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = call %bound_method.loc9_38.2(%n.ref)
// CHECK:STDOUT:     %.loc9_39.2: ref @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc9_39.3: init @Outer.F.%T.as_type.loc9_11.1 (%T.as_type) = initialize_from %.loc9_38.1 to %.loc9_39.2
// CHECK:STDOUT:     %.loc9_39.4: init @Outer.F.%Inner (%Inner.e67) = class_init (%.loc9_39.3), %return
// CHECK:STDOUT:     %.loc9_40: init @Outer.F.%Inner (%Inner.e67) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.a20)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%T) [symbolic = %Outer (constants.%Outer.7ce)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Outer [symbolic = %ptr (constants.%ptr.1aa)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.c68)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Outer.as.Destroy.impl.Op.%ptr (%ptr.1aa)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.e38 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.e38 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.576 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc13_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_38: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.dcf) [concrete = constants.%Copy.facet.715]
// CHECK:STDOUT:   %.loc13_38: %Copy.type = converted %i32.loc13_35, %Copy.facet.loc13_38 [concrete = constants.%Copy.facet.715]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%Copy.facet.715) [concrete = constants.%Outer.205]
// CHECK:STDOUT:   %.loc13_39: %Outer.F.type.d27 = specific_constant @Outer.%Outer.F.decl, @Outer(constants.%Copy.facet.715) [concrete = constants.%Outer.F.833]
// CHECK:STDOUT:   %F.ref: %Outer.F.type.d27 = name_ref F, %.loc13_39 [concrete = constants.%Outer.F.833]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %Copy.facet.loc13_43.1: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.36a) [concrete = constants.%Copy.facet.912]
// CHECK:STDOUT:   %.loc13_43.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.1 [concrete = constants.%Copy.facet.912]
// CHECK:STDOUT:   %Copy.facet.loc13_43.2: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.36a) [concrete = constants.%Copy.facet.912]
// CHECK:STDOUT:   %.loc13_43.2: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_43.2 [concrete = constants.%Copy.facet.912]
// CHECK:STDOUT:   %Outer.F.specific_fn: <specific function> = specific_function %F.ref, @Outer.F(constants.%Copy.facet.715) [concrete = constants.%Outer.F.specific_fn]
// CHECK:STDOUT:   %.loc13_3: ref %Inner.576 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %Outer.F.call: init %Inner.576 = call %Outer.F.specific_fn(%.loc13_42.2) to %.loc13_3
// CHECK:STDOUT:   assign %c.var, %Outer.F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.576] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc13_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Copy.facet.loc13_19: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.dcf) [concrete = constants.%Copy.facet.715]
// CHECK:STDOUT:     %.loc13_19: %Copy.type = converted %i32.loc13_16, %Copy.facet.loc13_19 [concrete = constants.%Copy.facet.715]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%Copy.facet.715) [concrete = constants.%Outer.205]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.715) [concrete = constants.%Inner.576]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.576]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.576 = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.576 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.ba1 = name_ref n, @Inner.%.loc6_10 [concrete = @Inner.%.loc6_10]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%Inner.as.Destroy.impl.Op.9b6
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%Inner.as.Destroy.impl.Op.9b6, @Inner.as.Destroy.impl.Op(constants.%Copy.facet.715) [concrete = constants.%Inner.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_3: <bound method> = bound_method %c.var, %Inner.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.c3e = addr_of %c.var
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc13_3(%addr)
// CHECK:STDOUT:   return %.loc14_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.a20) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.a20
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.e67
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.0d7
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.f56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.a20) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T.a20
// CHECK:STDOUT:   %T.as_type.loc6_12.2 => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.909
// CHECK:STDOUT:   %Inner => constants.%Inner.e67
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.d4d
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.744
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.09e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl(constants.%T.a20) {
// CHECK:STDOUT:   %T => constants.%T.a20
// CHECK:STDOUT:   %Inner => constants.%Inner.e67
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.30c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl.Op(constants.%T.a20) {
// CHECK:STDOUT:   %T => constants.%T.a20
// CHECK:STDOUT:   %Inner => constants.%Inner.e67
// CHECK:STDOUT:   %ptr => constants.%ptr.a76
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.669
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%T.a20) {
// CHECK:STDOUT:   %T => constants.%T.a20
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.47fbd7.1
// CHECK:STDOUT:   %Inner => constants.%Inner.e67
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.45d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl(constants.%T.a20) {
// CHECK:STDOUT:   %T => constants.%T.a20
// CHECK:STDOUT:   %Outer => constants.%Outer.7ce
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.c9f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl.Op(constants.%T.a20) {
// CHECK:STDOUT:   %T => constants.%T.a20
// CHECK:STDOUT:   %Outer => constants.%Outer.7ce
// CHECK:STDOUT:   %ptr => constants.%ptr.1aa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.715) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Copy.facet.715
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.576
// CHECK:STDOUT:   %Outer.F.type => constants.%Outer.F.type.d27
// CHECK:STDOUT:   %Outer.F => constants.%Outer.F.833
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.715) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Copy.facet.715
// CHECK:STDOUT:   %T.as_type.loc6_12.2 => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.576
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.ba1
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.033
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.54b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.F(constants.%Copy.facet.715) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.715
// CHECK:STDOUT:   %T.as_type.loc9_11.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %Inner => constants.%Inner.576
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.e38
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.54b
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.033
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.dcf
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.715
// CHECK:STDOUT:   %.loc9_38.2 => constants.%.8ee
// CHECK:STDOUT:   %impl.elem0.loc9_38.2 => constants.%Int.as.Copy.impl.Op.d19
// CHECK:STDOUT:   %specific_impl_fn.loc9_38.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl(constants.%Copy.facet.715) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.715
// CHECK:STDOUT:   %Inner => constants.%Inner.576
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.cad
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op.type => constants.%Inner.as.Destroy.impl.Op.type.7a4
// CHECK:STDOUT:   %Inner.as.Destroy.impl.Op => constants.%Inner.as.Destroy.impl.Op.9b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl.Op(constants.%Copy.facet.715) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.715
// CHECK:STDOUT:   %Inner => constants.%Inner.576
// CHECK:STDOUT:   %ptr => constants.%ptr.c3e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.55d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.392: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.770: %Inner.type.392 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.ea1: type = facet_access_type %Self.770 [symbolic]
// CHECK:STDOUT:   %pattern_type.81e: type = pattern_type %Self.as_type.ea1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Inner.F.type.0f3: type = fn_type @Inner.F, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.F.db9: %Inner.F.type.0f3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.115: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.95e: %Inner.assoc_type.115 = assoc_entity element0, @Inner.%Inner.F.decl [symbolic]
// CHECK:STDOUT:   %C.390: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.8fa: <witness> = require_complete_type %Inner.type.392 [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.b15: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e59: type = pattern_type %C.390 [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.77b: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Inner.impl.F.ed9: %C.as.Inner.impl.F.type.77b = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.9a3: %Inner.type.392 = facet_value %C.390, (%Inner.impl_witness.b15) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.674: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.306: type = ptr_type %C.390 [symbolic]
// CHECK:STDOUT:   %pattern_type.670: type = pattern_type %ptr.306 [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.d66: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.0c5: %C.as.Destroy.impl.Op.type.d66 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.a35: <witness> = impl_witness @Outer.%Destroy.impl_witness_table, @Outer.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %ptr.6ff: type = ptr_type %Outer.9d6 [symbolic]
// CHECK:STDOUT:   %pattern_type.07e: type = pattern_type %ptr.6ff [symbolic]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op, @Outer.as.Destroy.impl(%T) [symbolic]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: %Outer.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.930: <witness> = require_complete_type %C.390 [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.390, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.c18: %Inner.type.392 = facet_value %C.390, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.692: type = fn_type_with_self_type %Inner.F.type.0f3, %Inner.facet.c18 [symbolic]
// CHECK:STDOUT:   %impl.elem0.681: %.692 = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0.681, @Inner.F(%T, %Inner.facet.c18) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.545: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.52d: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.70f: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Self.493: %Inner.type.52d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Inner.F.type.86e: type = fn_type @Inner.F, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %Inner.F.11d: %Inner.F.type.86e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.assoc_type.215: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e80: %Inner.assoc_type.215 = assoc_entity element0, @Inner.%Inner.F.decl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.1dc: <witness> = impl_witness @D.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F.type: type = fn_type @D.as.Inner.impl.F [concrete]
// CHECK:STDOUT:   %D.as.Inner.impl.F: %D.as.Inner.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.edc: %Inner.type.52d = facet_value %D, (%Inner.impl_witness.1dc) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.73d: <witness> = impl_witness @D.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.type: type = fn_type @D.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %D.as.Destroy.impl.Op: %D.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.de9: type = pattern_type %C.70f [concrete]
// CHECK:STDOUT:   %C.val: %C.70f = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.9fa: <witness> = complete_type_witness %Inner.type.52d [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.47d: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.type.f11: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.e75: %C.as.Inner.impl.F.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.840: %Inner.type.52d = facet_value %C.70f, (%Inner.impl_witness.47d) [concrete]
// CHECK:STDOUT:   %.b10: type = fn_type_with_self_type %Inner.F.type.86e, %Inner.facet.840 [concrete]
// CHECK:STDOUT:   %C.as.Inner.impl.F.specific_fn: <specific function> = specific_function %C.as.Inner.impl.F.e75, @C.as.Inner.impl.F(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.82e: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type.97d: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.407: %C.as.Destroy.impl.Op.type.97d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.18f: type = ptr_type %C.70f [concrete]
// CHECK:STDOUT:   %pattern_type.7ba: type = pattern_type %ptr.18f [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %C.as.Destroy.impl.Op.407, @C.as.Destroy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
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
// CHECK:STDOUT:     %.Self: type = bind_symbolic_name .Self [symbolic_self = constants.%.Self.644]
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
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %Self.2: @Inner.%Inner.type (%Inner.type.392) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%T) [symbolic = %Inner.F.type (constants.%Inner.F.type.0f3)]
// CHECK:STDOUT:   %Inner.F: @Inner.%Inner.F.type (%Inner.F.type.0f3) = struct_value () [symbolic = %Inner.F (constants.%Inner.F.db9)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.115)]
// CHECK:STDOUT:   %assoc0.loc6_28.2: @Inner.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, %Inner.F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.95e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Inner.%Inner.type (%Inner.type.392) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:     %Inner.F.decl: @Inner.%Inner.F.type (%Inner.F.type.0f3) = fn_decl @Inner.F [symbolic = @Inner.%Inner.F (constants.%Inner.F.db9)] {
// CHECK:STDOUT:       %self.patt: @Inner.F.%pattern_type.loc6_10 (%pattern_type.81e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Inner.F.%pattern_type.loc6_10 (%pattern_type.81e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.F.%pattern_type.loc6_24 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.F.%pattern_type.loc6_24 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @Inner.F.%Self.as_type.loc6_16.1 (%Self.as_type.ea1) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %.loc6_16.3 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)] {
// CHECK:STDOUT:         %.loc6_16.2: @Inner.F.%Inner.type (%Inner.type.392) = specific_constant @Inner.%Self.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:         %Self.ref: @Inner.F.%Inner.type (%Inner.type.392) = name_ref Self, %.loc6_16.2 [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:         %Self.as_type.loc6_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:         %.loc6_16.3: type = converted %Self.ref, %Self.as_type.loc6_16.2 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Inner.F.%Self.as_type.loc6_16.1 (%Self.as_type.ea1) = bind_name self, %self.param
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
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.8fa)]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness @C.%Inner.impl_witness_table, @C.as.Inner.impl(%T) [symbolic = %Inner.impl_witness (constants.%Inner.impl_witness.b15)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type: type = fn_type @C.as.Inner.impl.F, @C.as.Inner.impl(%T) [symbolic = %C.as.Inner.impl.F.type (constants.%C.as.Inner.impl.F.type.77b)]
// CHECK:STDOUT:   %C.as.Inner.impl.F: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.77b) = struct_value () [symbolic = %C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.ed9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %C.as.Inner.impl.F.decl: @C.as.Inner.impl.%C.as.Inner.impl.F.type (%C.as.Inner.impl.F.type.77b) = fn_decl @C.as.Inner.impl.F [symbolic = @C.as.Inner.impl.%C.as.Inner.impl.F (constants.%C.as.Inner.impl.F.ed9)] {
// CHECK:STDOUT:       %self.patt: @C.as.Inner.impl.F.%pattern_type.loc11_12 (%pattern_type.e59) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_12 (%pattern_type.e59) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @C.as.Inner.impl.F.%pattern_type.loc11_23 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @C.as.Inner.impl.F.%pattern_type.loc11_23 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
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
// CHECK:STDOUT: generic impl @C.as.Destroy.impl(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @C.%Destroy.impl_witness_table, @C.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.674)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op, @C.as.Destroy.impl(%T) [symbolic = %C.as.Destroy.impl.Op.type (constants.%C.as.Destroy.impl.Op.type.d66)]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.d66) = struct_value () [symbolic = %C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.0c5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @C.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %C.as.Destroy.impl.Op.decl: @C.as.Destroy.impl.%C.as.Destroy.impl.Op.type (%C.as.Destroy.impl.Op.type.d66) = fn_decl @C.as.Destroy.impl.Op [symbolic = @C.as.Destroy.impl.%C.as.Destroy.impl.Op (constants.%C.as.Destroy.impl.Op.0c5)] {
// CHECK:STDOUT:       %self.patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.670) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @C.as.Destroy.impl.Op.%pattern_type (%pattern_type.670) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc9_11.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.306) = value_param call_param0
// CHECK:STDOUT:       %.loc9_11.2: type = splice_block %Self.ref [symbolic = %C (constants.%C.390)] {
// CHECK:STDOUT:         %.loc9_11.3: type = specific_constant constants.%C.390, @C(constants.%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc9_11.3 [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @C.as.Destroy.impl.Op.%ptr (%ptr.306) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @C.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Outer.as.Destroy.impl(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%T) [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Outer.%Destroy.impl_witness_table, @Outer.as.Destroy.impl(%T) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.a35)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op.type: type = fn_type @Outer.as.Destroy.impl.Op, @Outer.as.Destroy.impl(%T) [symbolic = %Outer.as.Destroy.impl.Op.type (constants.%Outer.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %Outer.as.Destroy.impl.Op: @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.type (%Outer.as.Destroy.impl.Op.type) = struct_value () [symbolic = %Outer.as.Destroy.impl.Op (constants.%Outer.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Outer.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     %Outer.as.Destroy.impl.Op.decl: @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.type (%Outer.as.Destroy.impl.Op.type) = fn_decl @Outer.as.Destroy.impl.Op [symbolic = @Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op (constants.%Outer.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @Outer.as.Destroy.impl.Op.%pattern_type (%pattern_type.07e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Outer.as.Destroy.impl.Op.%pattern_type (%pattern_type.07e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc4_23.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Outer.as.Destroy.impl.Op.%ptr (%ptr.6ff) = value_param call_param0
// CHECK:STDOUT:       %.loc4_23.2: type = splice_block %Self.ref [symbolic = %Outer (constants.%Outer.9d6)] {
// CHECK:STDOUT:         %.loc4_23.3: type = specific_constant constants.%Outer.9d6, @Outer(constants.%T) [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc4_23.3 [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Outer.as.Destroy.impl.Op.%ptr (%ptr.6ff) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Outer.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Outer.%Destroy.impl_witness
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
// CHECK:STDOUT: impl @D.as.Destroy.impl: @D.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:   %D.as.Destroy.impl.Op.decl: %D.as.Destroy.impl.Op.type = fn_decl @D.as.Destroy.impl.Op [concrete = constants.%D.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a94 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a94 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.19c = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %self: %ptr.19c = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %D.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @D.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_13.1)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_13.1) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.392)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.390)] {} {}
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Outer.9d6 [symbolic = @Outer.as.Destroy.impl.%Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:     impl_decl @Outer.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@Outer.as.Destroy.impl.%Outer.as.Destroy.impl.Op.decl), @Outer.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @Outer.as.Destroy.impl(constants.%T) [symbolic = @Outer.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.a35)]
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
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (@C.as.Inner.impl.%C.as.Inner.impl.F.decl), @C.as.Inner.impl [concrete]
// CHECK:STDOUT:     %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table, @C.as.Inner.impl(constants.%T) [symbolic = @C.as.Inner.impl.%Inner.impl_witness (constants.%Inner.impl_witness.b15)]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C.390 [symbolic = @C.as.Destroy.impl.%C (constants.%C.390)]
// CHECK:STDOUT:     impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @C.as.Destroy.impl(constants.%T) [symbolic = @C.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness.674)]
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
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (@D.as.Inner.impl.%D.as.Inner.impl.F.decl), @D.as.Inner.impl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.1dc]
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:   impl_decl @D.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@D.as.Destroy.impl.%D.as.Destroy.impl.Op.decl), @D.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.73d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.F(@Outer.%T.loc4_13.2: type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.392)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %Self: @Inner.F.%Inner.type (%Inner.type.392) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:   %Self.as_type.loc6_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:   %pattern_type.loc6_10: type = pattern_type %Self.as_type.loc6_16.1 [symbolic = %pattern_type.loc6_10 (constants.%pattern_type.81e)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %T [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.F.%Self.as_type.loc6_16.1 (%Self.as_type.ea1)) -> @Inner.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Inner.impl.F(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:   %pattern_type.loc11_12: type = pattern_type %C [symbolic = %pattern_type.loc11_12 (constants.%pattern_type.e59)]
// CHECK:STDOUT:   %pattern_type.loc11_23: type = pattern_type %T [symbolic = %pattern_type.loc11_23 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_23: <witness> = require_complete_type %T [symbolic = %require_complete.loc11_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc11_16: <witness> = require_complete_type %C [symbolic = %require_complete.loc11_16 (constants.%require_complete.930)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %require_complete.loc11_48: <witness> = require_complete_type %Inner.type [symbolic = %require_complete.loc11_48 (constants.%require_complete.8fa)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.115)]
// CHECK:STDOUT:   %assoc0: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, @Inner.%Inner.F.decl [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %Inner.F.type: type = fn_type @Inner.F, @Inner(%T) [symbolic = %Inner.F.type (constants.%Inner.F.type.0f3)]
// CHECK:STDOUT:   %Inner.facet: @C.as.Inner.impl.F.%Inner.type (%Inner.type.392) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.c18)]
// CHECK:STDOUT:   %.loc11_41: type = fn_type_with_self_type %Inner.F.type, %Inner.facet [symbolic = %.loc11_41 (constants.%.692)]
// CHECK:STDOUT:   %impl.elem0.loc11_41.2: @C.as.Inner.impl.F.%.loc11_41 (%.692) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0.681)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2: <specific function> = specific_impl_function %impl.elem0.loc11_41.2, @Inner.F(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Inner.impl.F.%C (%C.390)) -> @C.as.Inner.impl.F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @C.as.Inner.impl.F.%C (%C.390) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_43: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_43 [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:     %.loc11_48: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.115) = specific_constant @Inner.%assoc0.loc6_28.1, @Inner(constants.%T) [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:     %F.ref: @C.as.Inner.impl.F.%Inner.assoc_type (%Inner.assoc_type.115) = name_ref F, %.loc11_48 [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:     %impl.elem0.loc11_41.1: @C.as.Inner.impl.F.%.loc11_41 (%.692) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0.681)]
// CHECK:STDOUT:     %bound_method.loc11_41: <bound method> = bound_method %self.ref, %impl.elem0.loc11_41.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_41.1: <specific function> = specific_impl_function %impl.elem0.loc11_41.1, @Inner.F(constants.%T, constants.%Inner.facet.c18) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc11_52: <bound method> = bound_method %self.ref, %specific_impl_fn.loc11_41.1
// CHECK:STDOUT:     %.loc11_52: init @C.as.Inner.impl.F.%T (%T) = call %bound_method.loc11_52(%self.ref)
// CHECK:STDOUT:     %.loc11_53.1: @C.as.Inner.impl.F.%T (%T) = value_of_initializer %.loc11_52
// CHECK:STDOUT:     %.loc11_53.2: @C.as.Inner.impl.F.%T (%T) = converted %.loc11_52, %.loc11_53.1
// CHECK:STDOUT:     return %.loc11_53.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [symbolic = %ptr (constants.%ptr.306)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.670)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @C.as.Destroy.impl.Op.%ptr (%ptr.306)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(%T) [symbolic = %Outer (constants.%Outer.9d6)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Outer [symbolic = %ptr (constants.%ptr.6ff)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.07e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Outer.as.Destroy.impl.Op.%ptr (%ptr.6ff)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Inner.impl.F(%self.param: %D) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D.as.Destroy.impl.Op(%self.param: %ptr.19c) = "no_op";
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
// CHECK:STDOUT:   %c: ref %C.70f = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.70f = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc24: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc24: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc24: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc24: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:   %.loc24_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc24_23 [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:   %.loc24_29: %Inner.assoc_type.215 = specific_constant @Inner.%assoc0.loc6_28.1, @Inner(constants.%i32) [concrete = constants.%assoc0.e80]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.215 = name_ref F, %.loc24_29 [concrete = constants.%assoc0.e80]
// CHECK:STDOUT:   %impl.elem0: %.b10 = impl_witness_access constants.%Inner.impl_witness.47d, element0 [concrete = constants.%C.as.Inner.impl.F.e75]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @C.as.Inner.impl.F(constants.%i32) [concrete = constants.%C.as.Inner.impl.F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.70f = bind_value %c.ref
// CHECK:STDOUT:   %C.as.Inner.impl.F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %.loc24_34.1: %i32 = value_of_initializer %C.as.Inner.impl.F.call
// CHECK:STDOUT:   %.loc24_34.2: %i32 = converted %C.as.Inner.impl.F.call, %.loc24_34.1
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.bound: <bound method> = bound_method %c.var, constants.%C.as.Destroy.impl.Op.407
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%C.as.Destroy.impl.Op.407, @C.as.Destroy.impl.Op(constants.%i32) [concrete = constants.%C.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %c.var, %C.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.18f = addr_of %c.var
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc23(%addr)
// CHECK:STDOUT:   return %.loc24_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self.2 => constants.%Self.770
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.0f3
// CHECK:STDOUT:   %Inner.F => constants.%Inner.F.db9
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.115
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.95e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Self.770) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Self.770
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%Self.as_type.ea1
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.81e
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8fa
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.b15
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.77b
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.ed9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc11_12 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc11_23 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Inner.facet.9a3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Inner.facet.9a3
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.674
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %ptr => constants.%ptr.306
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.670
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer => constants.%Outer.9d6
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.a35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl.Op(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Outer => constants.%Outer.9d6
// CHECK:STDOUT:   %ptr => constants.%ptr.6ff
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.07e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%T, constants.%Inner.facet.c18) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Inner.facet.c18
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %Self.2 => constants.%Self.493
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
// CHECK:STDOUT: specific @Inner.F(constants.%i32, constants.%Inner.facet.edc) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %Self => constants.%Inner.facet.edc
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.510
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Inner.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9fa
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.47d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Inner.impl.F.type => constants.%C.as.Inner.impl.F.type.f11
// CHECK:STDOUT:   %C.as.Inner.impl.F => constants.%C.as.Inner.impl.F.e75
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
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %require_complete.loc11_48 => constants.%complete_type.9fa
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.215
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e80
// CHECK:STDOUT:   %Inner.lookup_impl_witness => constants.%Inner.impl_witness.47d
// CHECK:STDOUT:   %Inner.F.type => constants.%Inner.F.type.86e
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.840
// CHECK:STDOUT:   %.loc11_41 => constants.%.b10
// CHECK:STDOUT:   %impl.elem0.loc11_41.2 => constants.%C.as.Inner.impl.F.e75
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2 => constants.%C.as.Inner.impl.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.82e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type => constants.%C.as.Destroy.impl.Op.type.97d
// CHECK:STDOUT:   %C.as.Destroy.impl.Op => constants.%C.as.Destroy.impl.Op.407
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.as.Destroy.impl.Op(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT:   %ptr => constants.%ptr.18f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ba
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.F(constants.%i32, constants.%Inner.facet.840) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %Self => constants.%Inner.facet.840
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.70f
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.de9
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
