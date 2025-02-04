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
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+14]]:23: error: cannot implicitly convert from `<associated entity in J>` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+11]]:23: note: type `<associated entity in J>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+7]]:29: error: cannot implicitly convert from `<associated entity in J>` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+4]]:29: note: type `<associated entity in J>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR:
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

// --- fail_todo_use_non-type_in_function.carbon
library "[[@TEST_NAME]]";

interface M {
  let Z:! {.b: {}};
  fn G() -> {};
}

impl () as M where .Z = {.b = {}} {
  fn G() -> {} {
    // CHECK:STDERR: fail_todo_use_non-type_in_function.carbon:[[@LINE+7]]:13: error: cannot convert from `type` to `M` with `as` [ExplicitAsConversionFailure]
    // CHECK:STDERR:     return (Self as M).Z.b;
    // CHECK:STDERR:             ^~~~~~~~~
    // CHECK:STDERR: fail_todo_use_non-type_in_function.carbon:[[@LINE+4]]:13: note: type `type` does not implement interface `Core.As(M)` [MissingImplInMemberAccessNote]
    // CHECK:STDERR:     return (Self as M).Z.b;
    // CHECK:STDERR:             ^~~~~~~~~
    // CHECK:STDERR:
    return (Self as M).Z.b;
  }
}

// --- fail_todo_associated_int_in_array.carbon
library "[[@TEST_NAME]]";

interface I {
  let N:! i32;
  // CHECK:STDERR: fail_todo_associated_int_in_array.carbon:[[@LINE+7]]:32: error: cannot implicitly convert from `<associated entity in I>` to `Core.IntLiteral` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self]() -> [bool; N];
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: fail_todo_associated_int_in_array.carbon:[[@LINE+4]]:32: note: type `<associated entity in I>` does not implement interface `Core.ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self]() -> [bool; N];
  // CHECK:STDERR:                                ^
  // CHECK:STDERR:
  fn F[self: Self]() -> [bool; N];
}

impl () as I where .N = 2 {
  fn F[self: Self]() -> [bool; 2] { return (true, false); }
}

// CHECK:STDOUT: --- fail_todo_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [template]
// CHECK:STDOUT:   %assoc0.1e6: %J.assoc_type = assoc_entity element0, @J.%U [template]
// CHECK:STDOUT:   %Self.as_type.3df: type = facet_access_type %Self.ccd [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type.c14: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.b71: %F.type.c14 = struct_value () [template]
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, @J.%F.decl [template]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %J.facet.ad0: %J.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %J_where.type.800: type = facet_type <@J where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness.f71: <witness> = impl_witness (%empty_struct_type, <error>) [template]
// CHECK:STDOUT:   %F.type.159: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.59d: %F.type.159 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.550: %J.type = facet_value %empty_tuple.type, %impl_witness.f71 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %J_where.type.2f6: type = facet_type <@J where %impl.elem0 = %C> [template]
// CHECK:STDOUT:   %impl_witness.c2b: <witness> = impl_witness (%C, <error>) [template]
// CHECK:STDOUT:   %F.type.01a: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.686: %F.type.01a = struct_value () [template]
// CHECK:STDOUT:   %J.facet.38d: %J.type = facet_value %C, %impl_witness.c2b [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %.loc22_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_7.2: type = converted %.loc22_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [template = constants.%assoc0.1e6]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc22_20: type = converted %.Self.ref, %.Self.as_type [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc22_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_26.2: type = converted %.loc22_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc22_14: type = where_expr %.Self [template = constants.%J_where.type.800] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc22_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc22: <witness> = impl_witness (constants.%empty_struct_type, <error>) [template = constants.%impl_witness.f71]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref.loc31_6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [template = constants.%assoc0.1e6]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc31_19: type = converted %.Self.ref, %.Self.as_type [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref.loc31_24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc31_13: type = where_expr %.Self [template = constants.%J_where.type.2f6] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc31_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (constants.%C, <error>) [template = constants.%impl_witness.c2b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [template] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [template = constants.%assoc0.1e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.c14 = fn_decl @F.1 [template = constants.%F.b71] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.3df) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.3df) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc19_29: %J.assoc_type = name_ref U, @U.%assoc0 [template = constants.%assoc0.1e6]
// CHECK:STDOUT:     %.loc19_29: type = converted %U.ref.loc19_29, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.3df) = value_param runtime_param0
// CHECK:STDOUT:     %.loc19_14.1: type = splice_block %.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.3df)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:       %Self.as_type.loc19_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:       %.loc19_14.2: type = converted %Self.ref, %Self.as_type.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.3df) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: <error> = value_param runtime_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [template = <error>] {
// CHECK:STDOUT:       %U.ref.loc19_23: %J.assoc_type = name_ref U, @U.%assoc0 [template = constants.%assoc0.1e6]
// CHECK:STDOUT:       %.loc19_23: type = converted %U.ref.loc19_23, <error> [template = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %J.assoc_type = assoc_entity element1, %F.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc22_7.2 as %.loc22_14 {
// CHECK:STDOUT:   %F.decl: %F.type.159 = fn_decl @F.2 [template = constants.%F.59d] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: %empty_struct_type = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %empty_struct_type = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc23_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc23_31.2: type = converted %.loc23_31.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.1.%.loc22_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %empty_struct_type = value_param runtime_param1
// CHECK:STDOUT:     %.loc23_24.1: type = splice_block %.loc23_24.3 [template = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc23_24.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc23_24.3: type = converted %.loc23_24.2, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %empty_struct_type = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %C.ref.loc31_6 as %.loc31_13 {
// CHECK:STDOUT:   %F.decl: %F.type.01a = fn_decl @F.3 [template = constants.%F.686] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: %C = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %C = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc32_29: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%C.ref.loc31_6 [template = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %C.ref.loc32_23: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %u: %C = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc28_10: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc28_11: type = converted %.loc28_10, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   adapt_decl %.loc28_11 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.ccd)]
// CHECK:STDOUT:   %Self.as_type.loc19_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.3df)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.3df)](%u.param_patt: <error>) -> <error>;
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
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.ccd) {
// CHECK:STDOUT:   %Self => constants.%Self.ccd
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%Self.as_type.3df
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet.ad0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.550) {
// CHECK:STDOUT:   %Self => constants.%J.facet.550
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.38d) {
// CHECK:STDOUT:   %Self => constants.%J.facet.38d
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [template]
// CHECK:STDOUT:   %Self.bcc: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type %M.type [template]
// CHECK:STDOUT:   %assoc0.e3d: %M.assoc_type = assoc_entity element0, @M.%Z [template]
// CHECK:STDOUT:   %G.type.020: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.91c: %G.type.020 = struct_value () [template]
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, @M.%G.decl [template]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %M.facet.ba5: %M.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [template]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct, @impl.%G.decl) [template]
// CHECK:STDOUT:   %G.type.aa2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.816: %G.type.aa2 = struct_value () [template]
// CHECK:STDOUT:   %M.facet.940: %M.type = facet_value %empty_tuple.type, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [template = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc8_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_7.2: type = converted %.loc8_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [template = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %M.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Z.ref: %M.assoc_type = name_ref Z, @Z.%assoc0 [template = constants.%assoc0.e3d]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.347 = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_33.1: %struct_type.b.347 = struct_literal (%.loc8_32)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_33.2: %empty_struct_type = converted %.loc8_32, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.b.347 = struct_value (%.loc8_33.2) [template = constants.%struct]
// CHECK:STDOUT:     %.loc8_33.3: %struct_type.b.347 = converted %.loc8_33.1, %struct [template = constants.%struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [template = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_33.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%struct, @impl.%G.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.bcc]
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl @Z [template] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%Z [template = constants.%assoc0.e3d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type.020 = fn_decl @G.1 [template = constants.%G.91c] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %M.assoc_type = assoc_entity element1, %G.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = @Z.%assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Z(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const Z:! %struct_type.b.347;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %G.decl: %G.type.aa2 = fn_decl @G.2 [template = constants.%G.816] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_14.2: type = converted %.loc9_14.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G.1(@M.%Self: %M.type) {
// CHECK:STDOUT:   fn() -> %empty_struct_type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.2() -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, @impl.%.loc8_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %M.ref: type = name_ref M, file.%M.decl [template = constants.%M.type]
// CHECK:STDOUT:   %.loc17: %M.type = converted %Self.ref, <error> [template = <error>]
// CHECK:STDOUT:   %Z.ref: <error> = name_ref Z, <error> [template = <error>]
// CHECK:STDOUT:   %b.ref: <error> = name_ref b, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%Self.bcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%Self.bcc) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Z(constants.%M.facet.ba5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet.940) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [template]
// CHECK:STDOUT:   %assoc0.73c: %I.assoc_type = assoc_entity element0, @I.%N [template]
// CHECK:STDOUT:   %Self.as_type.b70: type = facet_access_type %Self.826 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.bc6: %F.type.cf0 = struct_value () [template]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%F.decl [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %I.facet.1f7: %I.type = facet_value %.Self.as_type, %.Self.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [template]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [template]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [template]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %int_2.ef8> [template]
// CHECK:STDOUT:   %impl_witness.2c9: <witness> = impl_witness (%int_2.ef8, <error>) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2.ecc, bool [template]
// CHECK:STDOUT:   %F.type.9f6: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.572: %F.type.9f6 = struct_value () [template]
// CHECK:STDOUT:   %I.facet.7b2: %I.type = facet_value %empty_tuple.type, %impl_witness.2c9 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (bool, bool) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%true, %false) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.44 [template] {} {
// CHECK:STDOUT:     %.loc15_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc15_7.2: type = converted %.loc15_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %N.ref: %I.assoc_type = name_ref N, @N.%assoc0 [template = constants.%assoc0.73c]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc15_20: type = converted %.Self.ref, %.Self.as_type [symbolic = constants.%.Self.as_type]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc15_20: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc15_25: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %int_2, %impl.elem0.loc15_25 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %bound_method, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %specific_fn(%int_2) [template = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc15_25.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc15_25.2: %i32 = converted %int_2, %.loc15_25.1 [template = constants.%int_2.ef8]
// CHECK:STDOUT:     %.loc15_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc15_20, %.loc15_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%int_2.ef8, <error>) [template = constants.%impl_witness.2c9]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl @N [template] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%N [template = constants.%assoc0.73c]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.cf0 = fn_decl @F.1 [template = constants.%F.bc6] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %N.ref: %I.assoc_type = name_ref N, @N.%assoc0 [template = constants.%assoc0.73c]
// CHECK:STDOUT:     %.loc12_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc12_26.2: type = converted %bool.make_type, %.loc12_26.1 [template = bool]
// CHECK:STDOUT:     %.loc12_32: Core.IntLiteral = converted %N.ref, <error> [template = <error>]
// CHECK:STDOUT:     %array_type: type = array_type <error>, bool [template = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = value_param runtime_param0
// CHECK:STDOUT:     %.loc12_14.1: type = splice_block %.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:       %Self.as_type.loc12_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:       %.loc12_14.2: type = converted %Self.ref, %Self.as_type.loc12_14.2 [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %F.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = @N.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%N, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @N(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const N:! %i32;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.44: %.loc15_7.2 as %.loc15_14 {
// CHECK:STDOUT:   %F.decl: %F.type.9f6 = fn_decl @F.2 [template = constants.%F.572] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc16_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc16_26.2: type = converted %bool.make_type, %.loc16_26.1 [template = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, bool [template = constants.%array_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.44.%.loc15_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %array_type = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %array_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@I.%Self: %I.type) {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.826)]
// CHECK:STDOUT:   %Self.as_type.loc12_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_14.1 (constants.%Self.as_type.b70)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc12_14.1 (%Self.as_type.b70)]() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type]() -> %return.param_patt: %array_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %.loc16_56.1: %tuple.type = tuple_literal (%true, %false)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc16_56.2: ref bool = array_index %return, %int_0
// CHECK:STDOUT:   %.loc16_56.3: init bool = initialize_from %true to %.loc16_56.2 [template = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc16_56.4: ref bool = array_index %return, %int_1
// CHECK:STDOUT:   %.loc16_56.5: init bool = initialize_from %false to %.loc16_56.4 [template = constants.%false]
// CHECK:STDOUT:   %.loc16_56.6: init %array_type = array_init (%.loc16_56.3, %.loc16_56.5) to %return [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57: init %array_type = converted %.loc16_56.1, %.loc16_56.6 [template = constants.%array]
// CHECK:STDOUT:   return %.loc16_57 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.826) {
// CHECK:STDOUT:   %Self => constants.%Self.826
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%Self.as_type.b70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%I.facet.1f7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet.7b2) {
// CHECK:STDOUT:   %Self => constants.%I.facet.7b2
// CHECK:STDOUT:   %Self.as_type.loc12_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
