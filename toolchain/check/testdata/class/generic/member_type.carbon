// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_type.carbon

// --- class.carbon

library "[[@TEST_NAME]]";

class Outer(T:! type) {
  //@dump-sem-ir-begin
  class Inner {
    var n: T;
  }
  //@dump-sem-ir-end

  fn F(n: T) -> Inner { return {.n = n}; }
}

fn Test() -> i32 {
  //@dump-sem-ir-begin
  var c: Outer(i32).Inner = Outer(i32).F(1);
  //@dump-sem-ir-end
  return c.n;
}

// --- interface.carbon

library "[[@TEST_NAME]]";

class Outer(T:! type) {
  //@dump-sem-ir-begin
  interface Inner {
    fn F[self: Self]() -> T;
  }
  //@dump-sem-ir-end

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
  //@dump-sem-ir-begin
  var c: Outer(i32).C = {};
  return c.(Outer(i32).Inner.F)();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.51b: type = class_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Inner.elem.310: type = unbound_element_type %Inner.51b, %T [symbolic]
// CHECK:STDOUT:   %struct_type.n.848: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.84b: <witness> = complete_type_witness %struct_type.n.848 [symbolic]
// CHECK:STDOUT:   %F.type.2ee: type = fn_type @F, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %F.384: %F.type.2ee = struct_value () [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Outer.545: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.721: type = class_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %F.type.9d7: type = fn_type @F, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %F.c88: %F.type.9d7 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.elem.6c2: type = unbound_element_type %Inner.721, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.033: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.54b: <witness> = complete_type_witness %struct_type.n.033 [concrete]
// CHECK:STDOUT:   %pattern_type.c22: type = pattern_type %Inner.721 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.c88, @F(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_13.2) [symbolic = %Inner (constants.%Inner.51b)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.51b)] {} {}
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.51b)]
// CHECK:STDOUT:   %Inner.elem: type = unbound_element_type %Inner, %T [symbolic = %Inner.elem (constants.%Inner.elem.310)]
// CHECK:STDOUT:   %struct_type.n.loc8_3.2: type = struct_type {.n: @Inner.%T (%T)} [symbolic = %struct_type.n.loc8_3.2 (constants.%struct_type.n.848)]
// CHECK:STDOUT:   %complete_type.loc8_3.2: <witness> = complete_type_witness %struct_type.n.loc8_3.2 [symbolic = %complete_type.loc8_3.2 (constants.%complete_type.84b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc7: @Inner.%Inner.elem (%Inner.elem.310) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %struct_type.n.loc8_3.1: type = struct_type {.n: %T} [symbolic = %struct_type.n.loc8_3.2 (constants.%struct_type.n.848)]
// CHECK:STDOUT:     %complete_type.loc8_3.1: <witness> = complete_type_witness %struct_type.n.loc8_3.1 [symbolic = %complete_type.loc8_3.2 (constants.%complete_type.84b)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc8_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.51b
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c22 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c22 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %Inner.721 = var %c.var_patt
// CHECK:STDOUT:   %Outer.ref.loc16_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc16_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc16_38: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:   %.loc16_39: %F.type.9d7 = specific_constant @Outer.%F.decl, @Outer(constants.%i32) [concrete = constants.%F.c88]
// CHECK:STDOUT:   %F.ref: %F.type.9d7 = name_ref F, %.loc16_39 [concrete = constants.%F.c88]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc16_3: ref %Inner.721 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc16_42.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_42.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc16_42.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_42.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc16_42.2: %i32 = converted %int_1, %.loc16_42.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %F.call: init %Inner.721 = call %F.specific_fn(%.loc16_42.2) to %.loc16_3
// CHECK:STDOUT:   assign %c.var, %F.call
// CHECK:STDOUT:   %.loc16_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.721] {
// CHECK:STDOUT:     %Outer.ref.loc16_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc16_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc16_19: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:     %.loc16_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.721]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc16_20.2 [concrete = constants.%Inner.721]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.721 = bind_name c, %c.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.51b
// CHECK:STDOUT:   %F.type => constants.%F.type.2ee
// CHECK:STDOUT:   %F => constants.%F.384
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Inner => constants.%Inner.51b
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.310
// CHECK:STDOUT:   %struct_type.n.loc8_3.2 => constants.%struct_type.n.848
// CHECK:STDOUT:   %complete_type.loc8_3.2 => constants.%complete_type.84b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner => constants.%Inner.721
// CHECK:STDOUT:   %F.type => constants.%F.type.9d7
// CHECK:STDOUT:   %F => constants.%F.c88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Inner => constants.%Inner.721
// CHECK:STDOUT:   %Inner.elem => constants.%Inner.elem.6c2
// CHECK:STDOUT:   %struct_type.n.loc8_3.2 => constants.%struct_type.n.033
// CHECK:STDOUT:   %complete_type.loc8_3.2 => constants.%complete_type.54b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.type.392: type = facet_type <@Inner, @Inner(%T)> [symbolic]
// CHECK:STDOUT:   %Self.770: %Inner.type.392 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.ea1: type = facet_access_type %Self.770 [symbolic]
// CHECK:STDOUT:   %pattern_type.81e: type = pattern_type %Self.as_type.ea1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type.0f3: type = fn_type @F.1, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %F.db9: %F.type.0f3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.115: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.95e: %Inner.assoc_type.115 = assoc_entity element0, @Inner.%F.decl [symbolic]
// CHECK:STDOUT:   %C.390: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.b15: <witness> = impl_witness @C.%Inner.impl_witness_table, @impl.eed(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e59: type = pattern_type %C.390 [symbolic]
// CHECK:STDOUT:   %Inner.facet.9a3: %Inner.type.392 = facet_value %C.390, (%Inner.impl_witness.b15) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.390, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.c18: %Inner.type.392 = facet_value %C.390, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.545: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.52d: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.70f: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Self.493: %Inner.type.52d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.86e: type = fn_type @F.1, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %F.11d: %F.type.86e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.assoc_type.215: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e80: %Inner.assoc_type.215 = assoc_entity element0, @Inner.%F.decl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.1dc: <witness> = impl_witness @D.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Inner.facet.edc: %Inner.type.52d = facet_value %D, (%Inner.impl_witness.1dc) [concrete]
// CHECK:STDOUT:   %pattern_type.de9: type = pattern_type %C.70f [concrete]
// CHECK:STDOUT:   %C.val: %C.70f = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.47d: <witness> = impl_witness @C.%Inner.impl_witness_table, @impl.eed(%i32) [concrete]
// CHECK:STDOUT:   %F.type.f11: type = fn_type @F.2, @impl.eed(%i32) [concrete]
// CHECK:STDOUT:   %F.e75: %F.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.840: %Inner.type.52d = facet_value %C.70f, (%Inner.impl_witness.47d) [concrete]
// CHECK:STDOUT:   %.b10: type = fn_type_with_self_type %F.type.86e, %Inner.facet.840 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.e75, @F.2(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Inner(%T) [symbolic = %F.type (constants.%F.type.0f3)]
// CHECK:STDOUT:   %F: @Inner.%F.type (%F.type.0f3) = struct_value () [symbolic = %F (constants.%F.db9)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.115)]
// CHECK:STDOUT:   %assoc0.loc7_28.2: @Inner.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc7_28.2 (constants.%assoc0.95e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %F.decl: @Inner.%F.type (%F.type.0f3) = fn_decl @F.1 [symbolic = @Inner.%F (constants.%F.db9)] {
// CHECK:STDOUT:       %self.patt: @F.1.%pattern_type.loc7_10 (%pattern_type.81e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.1.%pattern_type.loc7_10 (%pattern_type.81e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @F.1.%pattern_type.loc7_24 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.1.%pattern_type.loc7_24 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.1.%Self.as_type.loc7_16.1 (%Self.as_type.ea1) = value_param call_param0
// CHECK:STDOUT:       %.loc7_16.1: type = splice_block %.loc7_16.3 [symbolic = %Self.as_type.loc7_16.1 (constants.%Self.as_type.ea1)] {
// CHECK:STDOUT:         %.loc7_16.2: @F.1.%Inner.type (%Inner.type.392) = specific_constant @Inner.%Self.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:         %Self.ref: @F.1.%Inner.type (%Inner.type.392) = name_ref Self, %.loc7_16.2 [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:         %Self.as_type.loc7_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc7_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:         %.loc7_16.3: type = converted %Self.ref, %Self.as_type.loc7_16.2 [symbolic = %Self.as_type.loc7_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.1.%Self.as_type.loc7_16.1 (%Self.as_type.ea1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.1.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc7_28.1: @Inner.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc7_28.2 (constants.%assoc0.95e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc7_28.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_13.2)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.392)] {} {}
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT: generic fn @F.1(@Outer.%T.loc4_13.1: type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.392)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %Self: @F.1.%Inner.type (%Inner.type.392) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:   %Self.as_type.loc7_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc7_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:   %pattern_type.loc7_10: type = pattern_type %Self.as_type.loc7_16.1 [symbolic = %pattern_type.loc7_10 (constants.%pattern_type.81e)]
// CHECK:STDOUT:   %pattern_type.loc7_24: type = pattern_type %T [symbolic = %pattern_type.loc7_24 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.1.%Self.as_type.loc7_16.1 (%Self.as_type.ea1)) -> @F.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.de9 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.de9 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C.70f = var %c.var_patt
// CHECK:STDOUT:   %.loc26_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc26_26.2: init %C.70f = class_init (), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc26_3: init %C.70f = converted %.loc26_26.1, %.loc26_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc26_3
// CHECK:STDOUT:   %.loc26_20.1: type = splice_block %C.ref [concrete = constants.%C.70f] {
// CHECK:STDOUT:     %Outer.ref.loc26: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc26: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc26: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc26: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:     %.loc26_20.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%i32) [concrete = constants.%C.70f]
// CHECK:STDOUT:     %C.ref: type = name_ref C, %.loc26_20.2 [concrete = constants.%C.70f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %C.70f = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %C.70f = name_ref c, %c
// CHECK:STDOUT:   %Outer.ref.loc27: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc27: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:   %.loc27_23: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:   %Inner.ref: type = name_ref Inner, %.loc27_23 [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:   %.loc27_29: %Inner.assoc_type.215 = specific_constant @Inner.%assoc0.loc7_28.1, @Inner(constants.%i32) [concrete = constants.%assoc0.e80]
// CHECK:STDOUT:   %F.ref: %Inner.assoc_type.215 = name_ref F, %.loc27_29 [concrete = constants.%assoc0.e80]
// CHECK:STDOUT:   %impl.elem0: %.b10 = impl_witness_access constants.%Inner.impl_witness.47d, element0 [concrete = constants.%F.e75]
// CHECK:STDOUT:   %bound_method.loc27_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.2(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc27_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc27_10: %C.70f = bind_value %c.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %bound_method.loc27_33(%.loc27_10)
// CHECK:STDOUT:   %.loc27_34.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc27_34.2: %i32 = converted %F.call, %.loc27_34.1
// CHECK:STDOUT:   return %.loc27_34.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
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
// CHECK:STDOUT:   %F.type => constants.%F.type.0f3
// CHECK:STDOUT:   %F => constants.%F.db9
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.115
// CHECK:STDOUT:   %assoc0.loc7_28.2 => constants.%assoc0.95e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.770) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Self.770
// CHECK:STDOUT:   %Self.as_type.loc7_16.1 => constants.%Self.as_type.ea1
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.81e
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Inner.facet.9a3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Inner.facet.9a3
// CHECK:STDOUT:   %Self.as_type.loc7_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Inner.facet.c18) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Inner.facet.c18
// CHECK:STDOUT:   %Self.as_type.loc7_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%i32) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%i32
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
// CHECK:STDOUT:   %F.type => constants.%F.type.86e
// CHECK:STDOUT:   %F => constants.%F.11d
// CHECK:STDOUT:   %Inner.assoc_type => constants.%Inner.assoc_type.215
// CHECK:STDOUT:   %assoc0.loc7_28.2 => constants.%assoc0.e80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%i32, constants.%Inner.facet.edc) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %Self => constants.%Inner.facet.edc
// CHECK:STDOUT:   %Self.as_type.loc7_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.510
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%i32, constants.%Inner.facet.840) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %Self => constants.%Inner.facet.840
// CHECK:STDOUT:   %Self.as_type.loc7_16.1 => constants.%C.70f
// CHECK:STDOUT:   %pattern_type.loc7_10 => constants.%pattern_type.de9
// CHECK:STDOUT:   %pattern_type.loc7_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
