// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_todo_use_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_todo_use_assoc_const.carbon

// --- fail_todo_associated_type_in_signature.carbon
library "[[@TEST_NAME]]";

interface J {
  let U:! type;
  fn F[self: Self](u: U) -> U;
}

impl () as J where .U = {} {
  fn F[self: Self](u: {}) -> {} { return u; }
}

class C {
  // This allows the type to be copyable so it can be returned.
  adapt {};
}

impl C as J where .U = C {
  fn F[self: Self](u: C) -> C { return self; }
}

// --- use_non-type_in_function.carbon
library "[[@TEST_NAME]]";

interface M {
  let Z:! {.b: {}};
  fn G() -> {};
}

impl () as M where .Z = {.b = {}} {
  fn G() -> {} {
    return (Self as M).Z.b;
  }
}

// --- self_as_uses_correct_rewrite_constraint.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {}

interface M {
  let Z:! {.b: type};
  fn G() -> type;
}

impl C({}) as M where .Z = {.b = {}} {
  fn G() -> type {
    return (Self as M).Z.b;
  }
}

impl C(()) as M where .Z = {.b = ()} {
  fn G() -> type {
    return (Self as M).Z.b;
  }
}

// --- fail_todo_associated_int_in_array.carbon
library "[[@TEST_NAME]]";

interface I {
  let N:! i32;
  fn F[self: Self]() -> array(bool, N);
}

impl () as I where .N = 2 {
  fn F[self: Self]() -> array(bool, 2) { return (true, false); }
}

// CHECK:STDOUT: --- fail_todo_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Self.as_wit: <witness> = facet_access_witness %Self [symbolic]
// CHECK:STDOUT:   %J.facet.70f: %J.type = facet_value %Self.as_type, %Self.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0.f44: type = impl_witness_access %Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %J.facet.ad0: %J.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.767: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %J_where.type.800: type = facet_type <@J where %impl.elem0.767 = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %impl_witness.6de: <witness> = impl_witness (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %F.type.159: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.59d: %F.type.159 = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %J_where.type.2f6: type = facet_type <@J where %impl.elem0.767 = %C> [concrete]
// CHECK:STDOUT:   %impl_witness.61e: <witness> = impl_witness (%C) [concrete]
// CHECK:STDOUT:   %F.type.01a: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.686: %F.type.01a = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.2d9 [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0.767]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%J_where.type.800] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc8: <witness> = impl_witness (constants.%empty_struct_type) [concrete = constants.%impl_witness.6de]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.fb4 [concrete] {} {
// CHECK:STDOUT:     %C.ref.loc17_6: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc17_19: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0.767]
// CHECK:STDOUT:     %C.ref.loc17_24: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc17_13: type = where_expr %.Self [concrete = constants.%J_where.type.2f6] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc17_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (constants.%C) [concrete = constants.%impl_witness.61e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [concrete = constants.%F.b71] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.as_wit.loc5_29: <witness> = facet_access_witness @J.%Self [symbolic = %Self.as_wit.loc5_23.1 (constants.%Self.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc5_29: type = impl_witness_access %Self.as_wit.loc5_29, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.f44)]
// CHECK:STDOUT:     %U.ref.loc5_29: type = name_ref U, %impl.elem0.loc5_29 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.f44)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = value_param call_param1
// CHECK:STDOUT:     %.loc5_23: type = splice_block %U.ref.loc5_23 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.f44)] {
// CHECK:STDOUT:       %Self.as_wit.loc5_23.2: <witness> = facet_access_witness @J.%Self [symbolic = %Self.as_wit.loc5_23.1 (constants.%Self.as_wit)]
// CHECK:STDOUT:       %impl.elem0.loc5_23.2: type = impl_witness_access %Self.as_wit.loc5_23.2, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.f44)]
// CHECK:STDOUT:       %U.ref.loc5_23: type = name_ref U, %impl.elem0.loc5_23.2 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.f44)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = out_param call_param2
// CHECK:STDOUT:     %return: ref @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = <error>
// CHECK:STDOUT:   witness = (%U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2d9: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %F.decl: %F.type.159 = fn_decl @F.2 [concrete = constants.%F.59d] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: %empty_struct_type = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %empty_struct_type = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_31.2: type = converted %.loc9_31.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2d9.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %empty_struct_type = value_param call_param1
// CHECK:STDOUT:     %.loc9_24.1: type = splice_block %.loc9_24.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc9_24.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc9_24.3: type = converted %.loc9_24.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %empty_struct_type = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.fb4: %C.ref.loc17_6 as %.loc17_13 {
// CHECK:STDOUT:   %F.decl: %F.type.01a = fn_decl @F.3 [concrete = constants.%F.686] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %u.patt: %C = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %C = value_param_pattern %u.patt, call_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc18_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.fb4.%C.ref.loc17_6 [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param call_param1
// CHECK:STDOUT:     %C.ref.loc18_23: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %u: %C = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc14_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc14_11: type = converted %.loc14_10, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc14_11 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %Self.as_wit.loc5_23.1: <witness> = facet_access_witness %Self [symbolic = %Self.as_wit.loc5_23.1 (constants.%Self.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc5_23.1: type = impl_witness_access %Self.as_wit.loc5_23.1, element0 [symbolic = %impl.elem0.loc5_23.1 (constants.%impl.elem0.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)](%u.param_patt: @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44)) -> @F.1.%impl.elem0.loc5_23.1 (%impl.elem0.f44);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%u.param_patt: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %empty_struct_type = name_ref u, %u
// CHECK:STDOUT:   return %u.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %C](%u.param_patt: %C) -> %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   return %self.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.70f) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %Self.as_wit.loc5_23.1 => constants.%Self.as_wit
// CHECK:STDOUT:   %impl.elem0.loc5_23.1 => constants.%impl.elem0.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.ad0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type %M.type [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %M.facet.ba5: %M.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct) [concrete]
// CHECK:STDOUT:   %G.type.aa2: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.816: %G.type.aa2 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.837: %M.type = facet_value %empty_tuple.type, %impl_witness [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.347 = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_33.1: %struct_type.b.347 = struct_literal (%.loc8_32)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_33.2: %empty_struct_type = converted %.loc8_32, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.b.347 = struct_value (%.loc8_33.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_33.3: %struct_type.b.347 = converted %.loc8_33.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_33.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%struct) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = <error>
// CHECK:STDOUT:   witness = (%Z)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.347;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %G.decl: %G.type.aa2 = fn_decl @G.2 [concrete = constants.%G.816] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%empty_tuple.type, constants.%impl_witness [concrete = constants.%M.facet.837]
// CHECK:STDOUT:   %.loc10_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.837]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_18 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc10_23: type = converted %.loc10_18, %as_type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %as_wit: <witness> = facet_access_witness %.loc10_18 [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %as_wit, element0 [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc10_25: %empty_struct_type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc10_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.ba5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.837) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- self_as_uses_correct_rewrite_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.b.86f: type = struct_type {.b: type} [concrete]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type %M.type [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete]
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [concrete]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [concrete]
// CHECK:STDOUT:   %C.7a7: type = class_type @C, @C(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %M.facet.ba5: %M.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.0f3: %struct_type.b.86f = struct_value (%empty_struct_type) [concrete]
// CHECK:STDOUT:   %M_where.type.fd0: type = facet_type <@M where %impl.elem0 = %struct.0f3> [concrete]
// CHECK:STDOUT:   %impl_witness.a5c: <witness> = impl_witness (%struct.0f3) [concrete]
// CHECK:STDOUT:   %G.type.bce: type = fn_type @G.2 [concrete]
// CHECK:STDOUT:   %G.8af: %G.type.bce = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.df9: %M.type = facet_value %C.7a7, %impl_witness.a5c [concrete]
// CHECK:STDOUT:   %C.3a0: type = class_type @C, @C(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %struct_type.b.161: type = struct_type {.b: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %struct.c94: %struct_type.b.86f = struct_value (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %M_where.type.b49: type = facet_type <@M where %impl.elem0 = %struct.c94> [concrete]
// CHECK:STDOUT:   %impl_witness.ff1: <witness> = impl_witness (%struct.c94) [concrete]
// CHECK:STDOUT:   %G.type.875: type = fn_type @G.3 [concrete]
// CHECK:STDOUT:   %G.702: %G.type.875 = struct_value () [concrete]
// CHECK:STDOUT:   %M.facet.b99: %M.type = facet_value %C.3a0, %impl_witness.ff1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.b80 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc11_9: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_10: type = converted %.loc11_9, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_struct_type) [concrete = constants.%C.7a7]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc11_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc11_35: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc11_36.1: %struct_type.b.347 = struct_literal (%.loc11_35)
// CHECK:STDOUT:     %.loc11_36.2: type = converted %.loc11_35, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc11_36.2) [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc11_36.3: %struct_type.b.86f = converted %.loc11_36.1, %struct [concrete = constants.%struct.0f3]
// CHECK:STDOUT:     %.loc11_17: type = where_expr %.Self [concrete = constants.%M_where.type.fd0] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc11_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc11: <witness> = impl_witness (constants.%struct.0f3) [concrete = constants.%impl_witness.a5c]
// CHECK:STDOUT:   impl_decl @impl.be1 [concrete] {} {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.loc17_9: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_10: type = converted %.loc17_9, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%empty_tuple.type) [concrete = constants.%C.3a0]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc17_23: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.86f = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc17_35: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_36.1: %struct_type.b.161 = struct_literal (%.loc17_35)
// CHECK:STDOUT:     %.loc17_36.2: type = converted %.loc17_35, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct: %struct_type.b.86f = struct_value (%.loc17_36.2) [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc17_36.3: %struct_type.b.86f = converted %.loc17_36.1, %struct [concrete = constants.%struct.c94]
// CHECK:STDOUT:     %.loc17_17: type = where_expr %.Self [concrete = constants.%M_where.type.b49] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc17_36.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc17: <witness> = impl_witness (constants.%struct.c94) [concrete = constants.%impl_witness.ff1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Z: %struct_type.b.86f = assoc_const_decl @Z [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [concrete = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = <error>
// CHECK:STDOUT:   witness = (%Z)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.86f;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b80: %C as %.loc11_17 {
// CHECK:STDOUT:   %G.decl: %G.type.bce = fn_decl @G.2 [concrete = constants.%G.8af] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%impl_witness.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.be1: %C as %.loc17_17 {
// CHECK:STDOUT:   %G.decl: %G.type.875 = fn_decl @G.3 [concrete = constants.%G.702] {
// CHECK:STDOUT:     %return.patt: type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .M = <poisoned>
// CHECK:STDOUT:   witness = file.%impl_witness.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.b80.%C [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%C.7a7, constants.%impl_witness.a5c [concrete = constants.%M.facet.df9]
// CHECK:STDOUT:   %.loc13_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.df9]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc13_18 [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %.loc13_23: type = converted %.loc13_18, %as_type [concrete = constants.%C.7a7]
// CHECK:STDOUT:   %as_wit: <witness> = facet_access_witness %.loc13_18 [concrete = constants.%impl_witness.a5c]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access %as_wit, element0 [concrete = constants.%struct.0f3]
// CHECK:STDOUT:   %.loc13_25: type = struct_access %impl.elem0, element0 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   return %.loc13_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.3() -> type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.be1.%C [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value constants.%C.3a0, constants.%impl_witness.ff1 [concrete = constants.%M.facet.b99]
// CHECK:STDOUT:   %.loc19_18: %M.type = converted %Self.ref, %M.facet [concrete = constants.%M.facet.b99]
// CHECK:STDOUT:   %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc19_18 [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %.loc19_23: type = converted %.loc19_18, %as_type [concrete = constants.%C.3a0]
// CHECK:STDOUT:   %as_wit: <witness> = facet_access_witness %.loc19_18 [concrete = constants.%impl_witness.ff1]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.86f = impl_witness_access %as_wit, element0 [concrete = constants.%struct.c94]
// CHECK:STDOUT:   %.loc19_25: type = struct_access %impl.elem0, element0 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   return %.loc19_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_struct_type) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%empty_struct_type
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%empty_struct_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.ba5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.df9) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%empty_tuple.type
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.b99) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.73c: %I.assoc_type = assoc_entity element0, @I.%N [concrete]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.as_wit: <witness> = facet_access_witness %Self.826 [symbolic]
// CHECK:STDOUT:   %I.facet.648: %I.type = facet_value %Self.as_type.b70, %Self.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0.6f6: %i32 = impl_witness_access %Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.e25: %ImplicitAs.type.2fd = facet_value %i32, %impl_witness.023 [concrete]
// CHECK:STDOUT:   %.10e: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.e25 [concrete]
// CHECK:STDOUT:   %Convert.bound.6e1: <bound method> = bound_method %impl.elem0.6f6, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn.4f9: <specific function> = specific_function %Convert.bound.6e1, @Convert.3(%int_32) [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %Convert.specific_fn.4f9(%impl.elem0.6f6) [symbolic]
// CHECK:STDOUT:   %array_type.229: type = array_type %int.convert_checked, bool [symbolic]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic_self]
// CHECK:STDOUT:   %I.facet.1f7: %I.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fb7: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic_self]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f7f: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.f7f [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.787: <specific function> = specific_function %Convert.bound.ef9, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.fb7 = %int_2.ef8> [concrete]
// CHECK:STDOUT:   %impl_witness.af7: <witness> = impl_witness (%int_2.ef8) [concrete]
// CHECK:STDOUT:   %array_type.c9b: type = array_type %int_2.ecc, bool [concrete]
// CHECK:STDOUT:   %F.type.9f6: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.572: %F.type.9f6 = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type.c9b = tuple_value (%true, %false) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.581 [concrete] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %N.ref: %I.assoc_type = name_ref N, @N.%assoc0 [concrete = constants.%assoc0.73c]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic_self = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc8_20: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic_self = constants.%impl.elem0.fb7]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc8_25: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_2, %impl.elem0.loc8_25 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %bound_method, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.787]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %specific_fn(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_25.2: %i32 = converted %int_2, %.loc8_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_20, %.loc8_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%int_2.ef8) [concrete = constants.%impl_witness.af7]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%N [concrete = constants.%assoc0.73c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [concrete = constants.%F.bc6] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.1.%array_type.loc5_38.1 (%array_type.229) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.1.%array_type.loc5_38.1 (%array_type.229) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %Self.as_wit.loc5_37.2: <witness> = facet_access_witness @I.%Self [symbolic = %Self.as_wit.loc5_37.1 (constants.%Self.as_wit)]
// CHECK:STDOUT:     %impl.elem0.loc5_37.2: %i32 = impl_witness_access %Self.as_wit.loc5_37.2, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.6f6)]
// CHECK:STDOUT:     %N.ref: %i32 = name_ref N, %impl.elem0.loc5_37.2 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.6f6)]
// CHECK:STDOUT:     %.loc5_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc5_31.2: type = converted %bool.make_type, %.loc5_31.1 [concrete = bool]
// CHECK:STDOUT:     %impl.elem0.loc5_37.3: %.10e = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %N.ref, %impl.elem0.loc5_37.3 [symbolic = %Convert.bound (constants.%Convert.bound.6e1)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %bound_method, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn.4f9)]
// CHECK:STDOUT:     %int.convert_checked.loc5_37.2: init Core.IntLiteral = call %specific_fn(%N.ref) [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc5_37.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc5_37.2 [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc5_37.2: Core.IntLiteral = converted %N.ref, %.loc5_37.1 [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %array_type.loc5_38.2: type = array_type %.loc5_37.2, bool [symbolic = %array_type.loc5_38.1 (constants.%array_type.229)]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = value_param call_param0
// CHECK:STDOUT:     %.loc5_14.1: type = splice_block %.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc5_14.2: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref @F.1.%array_type.loc5_38.1 (%array_type.229) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.1.%array_type.loc5_38.1 (%array_type.229) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = @N.%assoc0
// CHECK:STDOUT:   .F = <error>
// CHECK:STDOUT:   witness = (%N)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @N(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const N:! %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.581: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %F.decl: %F.type.9f6 = fn_decl @F.2 [concrete = constants.%F.572] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %array_type.c9b = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type.c9b = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc9_31.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc9_31.2: type = converted %bool.make_type, %.loc9_31.1 [concrete = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, bool [concrete = constants.%array_type.c9b]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.581.%.loc8_7.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %array_type.c9b = out_param call_param1
// CHECK:STDOUT:     %return: ref %array_type.c9b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:   %Self.as_wit.loc5_37.1: <witness> = facet_access_witness %Self [symbolic = %Self.as_wit.loc5_37.1 (constants.%Self.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc5_37.1: %i32 = impl_witness_access %Self.as_wit.loc5_37.1, element0 [symbolic = %impl.elem0.loc5_37.1 (constants.%impl.elem0.6f6)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %impl.elem0.loc5_37.1, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound.6e1)]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.3(constants.%int_32) [symbolic = %Convert.specific_fn (constants.%Convert.specific_fn.4f9)]
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1: init Core.IntLiteral = call %Convert.specific_fn(%impl.elem0.loc5_37.1) [symbolic = %int.convert_checked.loc5_37.1 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc5_38.1: type = array_type %int.convert_checked.loc5_37.1, bool [symbolic = %array_type.loc5_38.1 (constants.%array_type.229)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type.b70)]() -> @F.1.%array_type.loc5_38.1 (%array_type.229);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type]() -> %return.param_patt: %array_type.c9b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_61.1: %tuple.type = tuple_literal (%true, %false)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_61.2: ref bool = array_index %return, %int_0
// CHECK:STDOUT:   %.loc9_61.3: init bool = initialize_from %true to %.loc9_61.2 [concrete = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_61.4: ref bool = array_index %return, %int_1
// CHECK:STDOUT:   %.loc9_61.5: init bool = initialize_from %false to %.loc9_61.4 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_61.6: init %array_type.c9b = array_init (%.loc9_61.3, %.loc9_61.5) to %return [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_62: init %array_type.c9b = converted %.loc9_61.1, %.loc9_61.6 [concrete = constants.%array]
// CHECK:STDOUT:   return %.loc9_62 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet.648) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.b70
// CHECK:STDOUT:   %Self.as_wit.loc5_37.1 => constants.%Self.as_wit
// CHECK:STDOUT:   %impl.elem0.loc5_37.1 => constants.%impl.elem0.6f6
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.6e1
// CHECK:STDOUT:   %Convert.specific_fn => constants.%Convert.specific_fn.4f9
// CHECK:STDOUT:   %int.convert_checked.loc5_37.1 => constants.%int.convert_checked
// CHECK:STDOUT:   %array_type.loc5_38.1 => constants.%array_type.229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet.1f7) {}
// CHECK:STDOUT:
