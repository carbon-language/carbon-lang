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
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+14]]:23: error: cannot implicitly convert from `<associated type in J>` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+11]]:23: note: type `<associated type in J>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+7]]:29: error: cannot implicitly convert from `<associated type in J>` to `type` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+4]]:29: note: type `<associated type in J>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
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
  // CHECK:STDERR: fail_todo_associated_int_in_array.carbon:[[@LINE+6]]:32: error: cannot implicitly convert from `<associated i32 in I>` to `Core.IntLiteral` [ImplicitAsConversionFailure]
  // CHECK:STDERR:   fn F[self: Self]() -> [bool; N];
  // CHECK:STDERR:                                ^
  // CHECK:STDERR: fail_todo_associated_int_in_array.carbon:[[@LINE+3]]:32: note: type `<associated i32 in I>` does not implement interface `Core.ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self]() -> [bool; N];
  // CHECK:STDERR:                                ^
  fn F[self: Self]() -> [bool; N];
}

impl () as I where .N = 2 {
  fn F[self: Self]() -> [bool; 2] { return (true, false); }
}

// CHECK:STDOUT: --- fail_todo_associated_type_in_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self.f9f: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %J.type, type [template]
// CHECK:STDOUT:   %assoc0.021: %assoc_type = assoc_entity element0, @J.%U [template]
// CHECK:STDOUT:   %Self.as_type.46d: type = facet_access_type %Self.f9f [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type.b65: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.83e: %F.type.b65 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %J.type, %F.type.b65 [template]
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, @J.%F.decl [template]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %J_where.type.800: type = facet_type <@J where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness.f71: <witness> = impl_witness (%empty_struct_type, <error>) [template]
// CHECK:STDOUT:   %F.type.31d: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.3db: %F.type.31d = struct_value () [template]
// CHECK:STDOUT:   %J.facet.550: %J.type = facet_value %empty_tuple.type, %impl_witness.f71 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %J_where.type.2f6: type = facet_type <@J where %impl.elem0 = %C> [template]
// CHECK:STDOUT:   %impl_witness.c2b: <witness> = impl_witness (%C, <error>) [template]
// CHECK:STDOUT:   %F.type.6c5: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.b69: %F.type.6c5 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.38d: %J.type = facet_value %C, %impl_witness.c2b [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:     %U.ref: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.021]
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
// CHECK:STDOUT:     %U.ref: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.021]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref.loc31_24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc31: type = where_expr %.Self [template = constants.%J_where.type.2f6] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc31_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc31: <witness> = impl_witness (constants.%C, <error>) [template = constants.%impl_witness.c2b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.f9f]
// CHECK:STDOUT:   %U: type = assoc_const_decl U [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %U [template = constants.%assoc0.021]
// CHECK:STDOUT:   %F.decl: %F.type.b65 = fn_decl @F.1 [template = constants.%F.83e] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.46d) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.46d) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc19_29: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.021]
// CHECK:STDOUT:     %.loc19_29: type = converted %U.ref.loc19_29, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.46d) = value_param runtime_param0
// CHECK:STDOUT:     %.loc19_14.1: type = splice_block %.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.46d)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.f9f)]
// CHECK:STDOUT:       %Self.as_type.loc19_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.46d)]
// CHECK:STDOUT:       %.loc19_14.2: type = converted %Self.ref, %Self.as_type.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.46d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.46d) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: <error> = value_param runtime_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [template = <error>] {
// CHECK:STDOUT:       %U.ref.loc19_23: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.021]
// CHECK:STDOUT:       %.loc19_23: type = converted %U.ref.loc19_23, <error> [template = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: <error> = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, %F.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = %assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%U, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.1: %.loc22_7.2 as %.loc22_14 {
// CHECK:STDOUT:   %F.decl: %F.type.31d = fn_decl @F.2 [template = constants.%F.3db] {
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
// CHECK:STDOUT: impl @impl.2: %C.ref.loc31_6 as %.loc31 {
// CHECK:STDOUT:   %F.decl: %F.type.6c5 = fn_decl @F.3 [template = constants.%F.b69] {
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.f9f)]
// CHECK:STDOUT:   %Self.as_type.loc19_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.46d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.46d)](%u.param_patt: <error>) -> <error>;
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
// CHECK:STDOUT: specific @F.1(constants.%Self.f9f) {
// CHECK:STDOUT:   %Self => constants.%Self.f9f
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%Self.as_type.46d
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Self.443: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %struct_type.b.347: type = struct_type {.b: %empty_struct_type} [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %M.type, %struct_type.b.347 [template]
// CHECK:STDOUT:   %assoc0.fbc: %assoc_type = assoc_entity element0, @M.%Z [template]
// CHECK:STDOUT:   %G.type.674: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.102: %G.type.674 = struct_value () [template]
// CHECK:STDOUT:   %G.assoc_type: type = assoc_entity_type %M.type, %G.type.674 [template]
// CHECK:STDOUT:   %assoc1: %G.assoc_type = assoc_entity element1, @M.%G.decl [template]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.347 = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %struct: %struct_type.b.347 = struct_value (%empty_struct) [template]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct, @impl.%G.decl) [template]
// CHECK:STDOUT:   %G.type.001: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.66a: %G.type.001 = struct_value () [template]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %empty_tuple.type, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .As = %import_ref.166
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
// CHECK:STDOUT:     %Z.ref: %assoc_type = name_ref Z, @M.%assoc0 [template = constants.%assoc0.fbc]
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
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.443]
// CHECK:STDOUT:   %Z: %struct_type.b.347 = assoc_const_decl Z [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %Z [template = constants.%assoc0.fbc]
// CHECK:STDOUT:   %G.decl: %G.type.674 = fn_decl @G.1 [template = constants.%G.102] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_14.2: type = converted %.loc5_14.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %G.assoc_type = assoc_entity element1, %G.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Z = %assoc0
// CHECK:STDOUT:   .G = %assoc1
// CHECK:STDOUT:   witness = (%Z, %G.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc8_7.2 as %.loc8_14 {
// CHECK:STDOUT:   %G.decl: %G.type.001 = fn_decl @G.2 [template = constants.%G.66a] {
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
// CHECK:STDOUT:
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
// CHECK:STDOUT: specific @G.1(constants.%Self.443) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.fb7: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, %i32 [template]
// CHECK:STDOUT:   %assoc0.285: %assoc_type = assoc_entity element0, @I.%N [template]
// CHECK:STDOUT:   %Self.as_type.e86: type = facet_access_type %Self.fb7 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.9df: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.478: %F.type.9df = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type.9df [template]
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, @I.%F.decl [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %int_2.166> [template]
// CHECK:STDOUT:   %impl_witness.184: <witness> = impl_witness (%int_2.166, <error>) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2.ecc, bool [template]
// CHECK:STDOUT:   %F.type.936: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.6b2: %F.type.936 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, %impl_witness.184 [template]
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
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
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
// CHECK:STDOUT:   impl_decl @impl.43 [template] {} {
// CHECK:STDOUT:     %.loc14_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_7.2: type = converted %.loc14_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %N.ref: %assoc_type = name_ref N, @I.%assoc0 [template = constants.%assoc0.285]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc14_20: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:     %impl.elem0.loc14_25: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0.loc14_25 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_2) [template = constants.%int_2.166]
// CHECK:STDOUT:     %.loc14_25.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_2.166]
// CHECK:STDOUT:     %.loc14_25.2: %i32 = converted %int_2, %.loc14_25.1 [template = constants.%int_2.166]
// CHECK:STDOUT:     %.loc14_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc14_20, %.loc14_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%int_2.166, <error>) [template = constants.%impl_witness.184]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fb7]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %N [template = constants.%assoc0.285]
// CHECK:STDOUT:   %F.decl: %F.type.9df = fn_decl @F.1 [template = constants.%F.478] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc11_14.1 (%Self.as_type.e86) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc11_14.1 (%Self.as_type.e86) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %N.ref: %assoc_type = name_ref N, @I.%assoc0 [template = constants.%assoc0.285]
// CHECK:STDOUT:     %.loc11_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_26.2: type = converted %bool.make_type, %.loc11_26.1 [template = bool]
// CHECK:STDOUT:     %.loc11_32: Core.IntLiteral = converted %N.ref, <error> [template = <error>]
// CHECK:STDOUT:     %array_type: type = array_type <error>, bool [template = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc11_14.1 (%Self.as_type.e86) = value_param runtime_param0
// CHECK:STDOUT:     %.loc11_14.1: type = splice_block %.loc11_14.2 [symbolic = %Self.as_type.loc11_14.1 (constants.%Self.as_type.e86)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.fb7)]
// CHECK:STDOUT:       %Self.as_type.loc11_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc11_14.1 (constants.%Self.as_type.e86)]
// CHECK:STDOUT:       %.loc11_14.2: type = converted %Self.ref, %Self.as_type.loc11_14.2 [symbolic = %Self.as_type.loc11_14.1 (constants.%Self.as_type.e86)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc11_14.1 (%Self.as_type.e86) = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param1
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, %F.decl [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%N, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.43: %.loc14_7.2 as %.loc14_14 {
// CHECK:STDOUT:   %F.decl: %F.type.936 = fn_decl @F.2 [template = constants.%F.6b2] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc15_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc15_26.2: type = converted %bool.make_type, %.loc15_26.1 [template = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, bool [template = constants.%array_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.43.%.loc14_7.2 [template = constants.%empty_tuple.type]
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.fb7)]
// CHECK:STDOUT:   %Self.as_type.loc11_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc11_14.1 (constants.%Self.as_type.e86)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc11_14.1 (%Self.as_type.e86)]() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type]() -> %return.param_patt: %array_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %.loc15_56.1: %tuple.type = tuple_literal (%true, %false)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc15_56.2: ref bool = array_index %return, %int_0
// CHECK:STDOUT:   %.loc15_56.3: init bool = initialize_from %true to %.loc15_56.2 [template = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc15_56.4: ref bool = array_index %return, %int_1
// CHECK:STDOUT:   %.loc15_56.5: init bool = initialize_from %false to %.loc15_56.4 [template = constants.%false]
// CHECK:STDOUT:   %.loc15_56.6: init %array_type = array_init (%.loc15_56.3, %.loc15_56.5) to %return [template = constants.%array]
// CHECK:STDOUT:   %.loc15_57: init %array_type = converted %.loc15_56.1, %.loc15_56.6 [template = constants.%array]
// CHECK:STDOUT:   return %.loc15_57 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.fb7) {
// CHECK:STDOUT:   %Self => constants.%Self.fb7
// CHECK:STDOUT:   %Self.as_type.loc11_14.1 => constants.%Self.as_type.e86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc11_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
