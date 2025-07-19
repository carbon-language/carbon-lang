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

class Outer(T:! type) {
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.51b: type = class_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Inner.elem.310: type = unbound_element_type %Inner.51b, %T [symbolic]
// CHECK:STDOUT:   %struct_type.n.848: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.84b: <witness> = complete_type_witness %struct_type.n.848 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.253: type = pattern_type %Inner.51b [symbolic]
// CHECK:STDOUT:   %F.type.2ee: type = fn_type @F, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %F.384: %F.type.2ee = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.561: <witness> = require_complete_type %Inner.51b [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a38: type = fn_type @Op.3, @Destroy.impl(%Inner.721) [concrete]
// CHECK:STDOUT:   %Op.27e: %Op.type.a38 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.416: type = ptr_type %Inner.721 [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.27e, @Op.3(%Inner.721) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
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
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T.loc4_13.2) [symbolic = %Inner (constants.%Inner.51b)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Outer(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type.2ee)]
// CHECK:STDOUT:   %F: @Outer.%F.type (%F.type.2ee) = struct_value () [symbolic = %F (constants.%F.384)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.51b)] {} {}
// CHECK:STDOUT:     %F.decl: @Outer.%F.type (%F.type.2ee) = fn_decl @F [symbolic = @Outer.%F (constants.%F.384)] {
// CHECK:STDOUT:       %n.patt: @F.%pattern_type.loc9_8 (%pattern_type.7dcd0a.1) = binding_pattern n [concrete]
// CHECK:STDOUT:       %n.param_patt: @F.%pattern_type.loc9_8 (%pattern_type.7dcd0a.1) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @F.%pattern_type.loc9_14 (%pattern_type.253) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.%pattern_type.loc9_14 (%pattern_type.253) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner (constants.%Inner.51b)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc9_17 [symbolic = %Inner (constants.%Inner.51b)]
// CHECK:STDOUT:       %n.param: @F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %n: @F.%T (%T) = bind_name n, %n.param
// CHECK:STDOUT:       %return.param: ref @F.%Inner (%Inner.51b) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.%Inner (%Inner.51b) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   %struct_type.n.loc7_3.2: type = struct_type {.n: @Inner.%T (%T)} [symbolic = %struct_type.n.loc7_3.2 (constants.%struct_type.n.848)]
// CHECK:STDOUT:   %complete_type.loc7_3.2: <witness> = complete_type_witness %struct_type.n.loc7_3.2 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.84b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc6: @Inner.%Inner.elem (%Inner.elem.310) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %struct_type.n.loc7_3.1: type = struct_type {.n: %T} [symbolic = %struct_type.n.loc7_3.2 (constants.%struct_type.n.848)]
// CHECK:STDOUT:     %complete_type.loc7_3.1: <witness> = complete_type_witness %struct_type.n.loc7_3.1 [symbolic = %complete_type.loc7_3.2 (constants.%complete_type.84b)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_3.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.51b
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .n = %.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Outer.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type.loc9_8: type = pattern_type %T [symbolic = %pattern_type.loc9_8 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(%T) [symbolic = %Inner (constants.%Inner.51b)]
// CHECK:STDOUT:   %pattern_type.loc9_14: type = pattern_type %Inner [symbolic = %pattern_type.loc9_14 (constants.%pattern_type.253)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14: <witness> = require_complete_type %Inner [symbolic = %require_complete.loc9_14 (constants.%require_complete.561)]
// CHECK:STDOUT:   %require_complete.loc9_9: <witness> = require_complete_type %T [symbolic = %require_complete.loc9_9 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @F.%T (%T)} [symbolic = %struct_type.n (constants.%struct_type.n.848)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @F.%T (%T)) -> %return.param: @F.%Inner (%Inner.51b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @F.%T (%T) = name_ref n, %n
// CHECK:STDOUT:     %.loc9_39.1: @F.%struct_type.n (%struct_type.n.848) = struct_literal (%n.ref)
// CHECK:STDOUT:     %.loc9_39.2: ref @F.%T (%T) = class_element_access %return, element0
// CHECK:STDOUT:     %.loc9_39.3: init @F.%T (%T) = initialize_from %n.ref to %.loc9_39.2
// CHECK:STDOUT:     %.loc9_39.4: init @F.%Inner (%Inner.51b) = class_init (%.loc9_39.3), %return
// CHECK:STDOUT:     %.loc9_40: init @F.%Inner (%Inner.51b) = converted %.loc9_39.1, %.loc9_39.4
// CHECK:STDOUT:     return %.loc9_40 to %return
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
// CHECK:STDOUT:   %Outer.ref.loc13_29: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32.loc13_35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Outer.loc13_38: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:   %.loc13_39: %F.type.9d7 = specific_constant @Outer.%F.decl, @Outer(constants.%i32) [concrete = constants.%F.c88]
// CHECK:STDOUT:   %F.ref: %F.type.9d7 = name_ref F, %.loc13_39 [concrete = constants.%F.c88]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc13_3: ref %Inner.721 = splice_block %c.var {}
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc13_42.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_42.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc13_42.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_42.2: %i32 = converted %int_1, %.loc13_42.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %F.call: init %Inner.721 = call %F.specific_fn(%.loc13_42.2) to %.loc13_3
// CHECK:STDOUT:   assign %c.var, %F.call
// CHECK:STDOUT:   %.loc13_20.1: type = splice_block %Inner.ref [concrete = constants.%Inner.721] {
// CHECK:STDOUT:     %Outer.ref.loc13_10: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32.loc13_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer.loc13_19: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:     %.loc13_20.2: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.721]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc13_20.2 [concrete = constants.%Inner.721]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %Inner.721 = bind_name c, %c.var
// CHECK:STDOUT:   %c.ref: ref %Inner.721 = name_ref c, %c
// CHECK:STDOUT:   %n.ref: %Inner.elem.6c2 = name_ref n, @Inner.%.loc6 [concrete = @Inner.%.loc6]
// CHECK:STDOUT:   %.loc14_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc14_11.2: %i32 = bind_value %.loc14_11.1
// CHECK:STDOUT:   %Op.bound.loc13_3.1: <bound method> = bound_method %.loc13_3, constants.%Op.27e
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.27e, @Op.3(constants.%Inner.721) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_3.1: <bound method> = bound_method %.loc13_3, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc13_3.1: %ptr.416 = addr_of %.loc13_3
// CHECK:STDOUT:   %no_op.loc13_3.1: init %empty_tuple.type = call %bound_method.loc13_3.1(%addr.loc13_3.1)
// CHECK:STDOUT:   %Op.bound.loc13_3.2: <bound method> = bound_method %c.var, constants.%Op.27e
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.27e, @Op.3(constants.%Inner.721) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_3.2: <bound method> = bound_method %c.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc13_3.2: %ptr.416 = addr_of %c.var
// CHECK:STDOUT:   %no_op.loc13_3.2: init %empty_tuple.type = call %bound_method.loc13_3.2(%addr.loc13_3.2)
// CHECK:STDOUT:   return %.loc14_11.2
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
// CHECK:STDOUT:   %struct_type.n.loc7_3.2 => constants.%struct_type.n.848
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.84b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:   %Inner => constants.%Inner.51b
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.253
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
// CHECK:STDOUT:   %struct_type.n.loc7_3.2 => constants.%struct_type.n.033
// CHECK:STDOUT:   %complete_type.loc7_3.2 => constants.%complete_type.54b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc9_8 => constants.%pattern_type.7ce
// CHECK:STDOUT:   %Inner => constants.%Inner.721
// CHECK:STDOUT:   %pattern_type.loc9_14 => constants.%pattern_type.c22
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_14 => constants.%complete_type.54b
// CHECK:STDOUT:   %require_complete.loc9_9 => constants.%complete_type.f8a
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n.033
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:   %F.type.0f3: type = fn_type @F.loc6, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %F.db9: %F.type.0f3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.assoc_type.115: type = assoc_entity_type @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.95e: %Inner.assoc_type.115 = assoc_entity element0, @Inner.%F.decl [symbolic]
// CHECK:STDOUT:   %C.390: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.8fa: <witness> = require_complete_type %Inner.type.392 [symbolic]
// CHECK:STDOUT:   %Inner.impl_witness.b15: <witness> = impl_witness @C.%Inner.impl_witness_table, @Inner.impl.eed(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.e59: type = pattern_type %C.390 [symbolic]
// CHECK:STDOUT:   %F.type.77b: type = fn_type @F.loc11, @Inner.impl.eed(%T) [symbolic]
// CHECK:STDOUT:   %F.ed9: %F.type.77b = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.facet.9a3: %Inner.type.392 = facet_value %C.390, (%Inner.impl_witness.b15) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.930: <witness> = require_complete_type %C.390 [symbolic]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C.390, @Inner, @Inner(%T) [symbolic]
// CHECK:STDOUT:   %Inner.facet.c18: %Inner.type.392 = facet_value %C.390, (%Inner.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.692: type = fn_type_with_self_type %F.type.0f3, %Inner.facet.c18 [symbolic]
// CHECK:STDOUT:   %impl.elem0.681: %.692 = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0.681, @F.loc6(%T, %Inner.facet.c18) [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Outer.545: type = class_type @Outer, @Outer(%i32) [concrete]
// CHECK:STDOUT:   %Inner.type.52d: type = facet_type <@Inner, @Inner(%i32)> [concrete]
// CHECK:STDOUT:   %C.70f: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %Self.493: %Inner.type.52d = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.86e: type = fn_type @F.loc6, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %F.11d: %F.type.86e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.assoc_type.215: type = assoc_entity_type @Inner, @Inner(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.e80: %Inner.assoc_type.215 = assoc_entity element0, @Inner.%F.decl [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.1dc: <witness> = impl_witness @D.%Inner.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.a78: type = fn_type @F.loc18 [concrete]
// CHECK:STDOUT:   %F.a24: %F.type.a78 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.edc: %Inner.type.52d = facet_value %D, (%Inner.impl_witness.1dc) [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.de9: type = pattern_type %C.70f [concrete]
// CHECK:STDOUT:   %C.val: %C.70f = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.9fa: <witness> = complete_type_witness %Inner.type.52d [concrete]
// CHECK:STDOUT:   %Inner.impl_witness.47d: <witness> = impl_witness @C.%Inner.impl_witness_table, @Inner.impl.eed(%i32) [concrete]
// CHECK:STDOUT:   %F.type.f11: type = fn_type @F.loc11, @Inner.impl.eed(%i32) [concrete]
// CHECK:STDOUT:   %F.e75: %F.type.f11 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.facet.840: %Inner.type.52d = facet_value %C.70f, (%Inner.impl_witness.47d) [concrete]
// CHECK:STDOUT:   %.b10: type = fn_type_with_self_type %F.type.86e, %Inner.facet.840 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.e75, @F.loc11(%i32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.b66: type = fn_type @Op.3, @Destroy.impl(%C.70f) [concrete]
// CHECK:STDOUT:   %Op.369: %Op.type.b66 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.18f: type = ptr_type %C.70f [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.369, @Op.3(%C.70f) [concrete]
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
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
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
// CHECK:STDOUT: generic interface @Inner(@Outer.%T.loc4_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %Self.2: @Inner.%Inner.type (%Inner.type.392) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc6, @Inner(%T) [symbolic = %F.type (constants.%F.type.0f3)]
// CHECK:STDOUT:   %F: @Inner.%F.type (%F.type.0f3) = struct_value () [symbolic = %F (constants.%F.db9)]
// CHECK:STDOUT:   %Inner.assoc_type: type = assoc_entity_type @Inner, @Inner(%T) [symbolic = %Inner.assoc_type (constants.%Inner.assoc_type.115)]
// CHECK:STDOUT:   %assoc0.loc6_28.2: @Inner.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.95e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @Inner.%Inner.type (%Inner.type.392) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.770)]
// CHECK:STDOUT:     %F.decl: @Inner.%F.type (%F.type.0f3) = fn_decl @F.loc6 [symbolic = @Inner.%F (constants.%F.db9)] {
// CHECK:STDOUT:       %self.patt: @F.loc6.%pattern_type.loc6_10 (%pattern_type.81e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc6.%pattern_type.loc6_10 (%pattern_type.81e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @F.loc6.%pattern_type.loc6_24 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.loc6.%pattern_type.loc6_24 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.loc6.%Self.as_type.loc6_16.1 (%Self.as_type.ea1) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %.loc6_16.3 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)] {
// CHECK:STDOUT:         %.loc6_16.2: @F.loc6.%Inner.type (%Inner.type.392) = specific_constant @Inner.%Self.1, @Inner(constants.%T) [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:         %Self.ref: @F.loc6.%Inner.type (%Inner.type.392) = name_ref Self, %.loc6_16.2 [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:         %Self.as_type.loc6_16.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:         %.loc6_16.3: type = converted %Self.ref, %Self.as_type.loc6_16.2 [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc6.%Self.as_type.loc6_16.1 (%Self.as_type.ea1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.loc6.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.loc6.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc6_28.1: @Inner.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc6_28.2 (constants.%assoc0.95e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc6_28.1
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Inner.impl.eed(@Outer.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Inner.type [symbolic = %require_complete (constants.%require_complete.8fa)]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness @C.%Inner.impl_witness_table, @Inner.impl.eed(%T) [symbolic = %Inner.impl_witness (constants.%Inner.impl_witness.b15)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc11, @Inner.impl.eed(%T) [symbolic = %F.type (constants.%F.type.77b)]
// CHECK:STDOUT:   %F: @Inner.impl.eed.%F.type (%F.type.77b) = struct_value () [symbolic = %F (constants.%F.ed9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %Inner.ref {
// CHECK:STDOUT:     %F.decl: @Inner.impl.eed.%F.type (%F.type.77b) = fn_decl @F.loc11 [symbolic = @Inner.impl.eed.%F (constants.%F.ed9)] {
// CHECK:STDOUT:       %self.patt: @F.loc11.%pattern_type.loc11_12 (%pattern_type.e59) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc11.%pattern_type.loc11_12 (%pattern_type.e59) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @F.loc11.%pattern_type.loc11_23 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @F.loc11.%pattern_type.loc11_23 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Outer.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.loc11.%C (%C.390) = value_param call_param0
// CHECK:STDOUT:       %.loc11_18.1: type = splice_block %C.ref [symbolic = %C (constants.%C.390)] {
// CHECK:STDOUT:         %.loc11_18.2: type = specific_constant @Outer.%C.decl, @Outer(constants.%T) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:         %C.ref: type = name_ref C, %.loc11_18.2 [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc11.%C (%C.390) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @F.loc11.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.loc11.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Inner = <poisoned>
// CHECK:STDOUT:     witness = @C.%Inner.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Inner.impl.a83: %Self.ref as %Inner.ref {
// CHECK:STDOUT:   %F.decl: %F.type.a78 = fn_decl @F.loc18 [concrete = constants.%F.a24] {
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
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @D.%Inner.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T.loc4_13.2)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_13.2) [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: type = interface_decl @Inner [symbolic = @Outer.%Inner.type (constants.%Inner.type.392)] {} {}
// CHECK:STDOUT:     %C.decl: type = class_decl @C [symbolic = @Outer.%C (constants.%C.390)] {} {}
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT: generic class @C(@Outer.%T.loc4_13.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @Inner.impl.eed [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C.390 [symbolic = %C (constants.%C.390)]
// CHECK:STDOUT:       %.loc10: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:       %Inner.ref: type = name_ref Inner, %.loc10 [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Inner.impl_witness_table = impl_witness_table (@Inner.impl.eed.%F.decl), @Inner.impl.eed [concrete]
// CHECK:STDOUT:     %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table, @Inner.impl.eed(constants.%T) [symbolic = @Inner.impl.eed.%Inner.impl_witness (constants.%Inner.impl_witness.b15)]
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
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
// CHECK:STDOUT:   impl_decl @Inner.impl.a83 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D [concrete = constants.%D]
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%i32) [concrete = constants.%Outer.545]
// CHECK:STDOUT:     %.loc17: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%i32) [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc17 [concrete = constants.%Inner.type.52d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Inner.impl_witness_table = impl_witness_table (@Inner.impl.a83.%F.decl), @Inner.impl.a83 [concrete]
// CHECK:STDOUT:   %Inner.impl_witness: <witness> = impl_witness %Inner.impl_witness_table [concrete = constants.%Inner.impl_witness.1dc]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .Outer = <poisoned>
// CHECK:STDOUT:   .D = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc6(@Outer.%T.loc4_13.1: type, @Inner.%Self.1: @Inner.%Inner.type (%Inner.type.392)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Inner.type: type = facet_type <@Inner, @Inner(%T)> [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:   %Self: @F.loc6.%Inner.type (%Inner.type.392) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.770)]
// CHECK:STDOUT:   %Self.as_type.loc6_16.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc6_16.1 (constants.%Self.as_type.ea1)]
// CHECK:STDOUT:   %pattern_type.loc6_10: type = pattern_type %Self.as_type.loc6_16.1 [symbolic = %pattern_type.loc6_10 (constants.%pattern_type.81e)]
// CHECK:STDOUT:   %pattern_type.loc6_24: type = pattern_type %T [symbolic = %pattern_type.loc6_24 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.loc6.%Self.as_type.loc6_16.1 (%Self.as_type.ea1)) -> @F.loc6.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc11(@Outer.%T.loc4_13.1: type) {
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
// CHECK:STDOUT:   %assoc0: @F.loc11.%Inner.assoc_type (%Inner.assoc_type.115) = assoc_entity element0, @Inner.%F.decl [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:   %Inner.lookup_impl_witness: <witness> = lookup_impl_witness %C, @Inner, @Inner(%T) [symbolic = %Inner.lookup_impl_witness (constants.%Inner.lookup_impl_witness)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc6, @Inner(%T) [symbolic = %F.type (constants.%F.type.0f3)]
// CHECK:STDOUT:   %Inner.facet: @F.loc11.%Inner.type (%Inner.type.392) = facet_value %C, (%Inner.lookup_impl_witness) [symbolic = %Inner.facet (constants.%Inner.facet.c18)]
// CHECK:STDOUT:   %.loc11_41: type = fn_type_with_self_type %F.type, %Inner.facet [symbolic = %.loc11_41 (constants.%.692)]
// CHECK:STDOUT:   %impl.elem0.loc11_41.2: @F.loc11.%.loc11_41 (%.692) = impl_witness_access %Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0.681)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2: <specific function> = specific_impl_function %impl.elem0.loc11_41.2, @F.loc6(%T, %Inner.facet) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @F.loc11.%C (%C.390)) -> @F.loc11.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @F.loc11.%C (%C.390) = name_ref self, %self
// CHECK:STDOUT:     %.loc11_43: type = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:     %Inner.ref: type = name_ref Inner, %.loc11_43 [symbolic = %Inner.type (constants.%Inner.type.392)]
// CHECK:STDOUT:     %.loc11_48: @F.loc11.%Inner.assoc_type (%Inner.assoc_type.115) = specific_constant @Inner.%assoc0.loc6_28.1, @Inner(constants.%T) [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:     %F.ref: @F.loc11.%Inner.assoc_type (%Inner.assoc_type.115) = name_ref F, %.loc11_48 [symbolic = %assoc0 (constants.%assoc0.95e)]
// CHECK:STDOUT:     %impl.elem0.loc11_41.1: @F.loc11.%.loc11_41 (%.692) = impl_witness_access constants.%Inner.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_41.2 (constants.%impl.elem0.681)]
// CHECK:STDOUT:     %bound_method.loc11_41: <bound method> = bound_method %self.ref, %impl.elem0.loc11_41.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_41.1: <specific function> = specific_impl_function %impl.elem0.loc11_41.1, @F.loc6(constants.%T, constants.%Inner.facet.c18) [symbolic = %specific_impl_fn.loc11_41.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc11_52: <bound method> = bound_method %self.ref, %specific_impl_fn.loc11_41.1
// CHECK:STDOUT:     %.loc11_52: init @F.loc11.%T (%T) = call %bound_method.loc11_52(%self.ref)
// CHECK:STDOUT:     %.loc11_53.1: @F.loc11.%T (%T) = value_of_initializer %.loc11_52
// CHECK:STDOUT:     %.loc11_53.2: @F.loc11.%T (%T) = converted %.loc11_52, %.loc11_53.1
// CHECK:STDOUT:     return %.loc11_53.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc18(%self.param: %D) -> %i32;
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
// CHECK:STDOUT:   %impl.elem0: %.b10 = impl_witness_access constants.%Inner.impl_witness.47d, element0 [concrete = constants.%F.e75]
// CHECK:STDOUT:   %bound_method.loc24_11: <bound method> = bound_method %c.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @F.loc11(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24_33: <bound method> = bound_method %c.ref, %specific_fn
// CHECK:STDOUT:   %.loc24_10: %C.70f = bind_value %c.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %bound_method.loc24_33(%.loc24_10)
// CHECK:STDOUT:   %.loc24_34.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc24_34.2: %i32 = converted %F.call, %.loc24_34.1
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %c.var, constants.%Op.369
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.369, @Op.3(constants.%C.70f) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23: <bound method> = bound_method %c.var, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.18f = addr_of %c.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc23(%addr)
// CHECK:STDOUT:   return %.loc24_34.2
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
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.95e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc6(constants.%T, constants.%Self.770) {
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
// CHECK:STDOUT: specific @Inner.impl.eed(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %require_complete => constants.%require_complete.8fa
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.b15
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.77b
// CHECK:STDOUT:   %F => constants.%F.ed9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc11(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %C => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc11_12 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc11_23 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc6(constants.%T, constants.%Inner.facet.9a3) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Inner.facet.9a3
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc6(constants.%T, constants.%Inner.facet.c18) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.392
// CHECK:STDOUT:   %Self => constants.%Inner.facet.c18
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.390
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.e59
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7dcd0a.1
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
// CHECK:STDOUT:   %assoc0.loc6_28.2 => constants.%assoc0.e80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc6(constants.%i32, constants.%Inner.facet.edc) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %Self => constants.%Inner.facet.edc
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.510
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.impl.eed(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %C => constants.%C.70f
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9fa
// CHECK:STDOUT:   %Inner.impl_witness => constants.%Inner.impl_witness.47d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.f11
// CHECK:STDOUT:   %F => constants.%F.e75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc11(constants.%i32) {
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
// CHECK:STDOUT:   %F.type => constants.%F.type.86e
// CHECK:STDOUT:   %Inner.facet => constants.%Inner.facet.840
// CHECK:STDOUT:   %.loc11_41 => constants.%.b10
// CHECK:STDOUT:   %impl.elem0.loc11_41.2 => constants.%F.e75
// CHECK:STDOUT:   %specific_impl_fn.loc11_41.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc6(constants.%i32, constants.%Inner.facet.840) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.52d
// CHECK:STDOUT:   %Self => constants.%Inner.facet.840
// CHECK:STDOUT:   %Self.as_type.loc6_16.1 => constants.%C.70f
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.de9
// CHECK:STDOUT:   %pattern_type.loc6_24 => constants.%pattern_type.7ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
