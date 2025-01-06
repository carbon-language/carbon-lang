// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/fail_todo_use_assoc_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/fail_todo_use_assoc_constant.carbon

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

class C {}

impl C as J where .U = C {
  // CHECK:STDERR: fail_todo_associated_type_in_signature.carbon:[[@LINE+4]]:40: error: cannot copy value of type `C` [CopyOfUncopyableType]
  // CHECK:STDERR:   fn F[self: Self](u: C) -> C { return self; }
  // CHECK:STDERR:                                        ^~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %Self.1: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %J.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @J.%U [template]
// CHECK:STDOUT:   %Self.as_type.1: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %J.type, %F.type.1 [template]
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, @J.%F.decl [template]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %J_where.type.1: type = facet_type <@J where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness.1: <witness> = impl_witness (%empty_struct_type, <error>) [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.1: %J.type = facet_value %empty_tuple.type, %impl_witness.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %J_where.type.2: type = facet_type <@J where %impl.elem0 = %C> [template]
// CHECK:STDOUT:   %impl_witness.2: <witness> = impl_witness (%C, <error>) [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %J.facet.2: %J.type = facet_value %C, %impl_witness.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .ImplicitAs = %import_ref.1
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
// CHECK:STDOUT:     %U.ref: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc22_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_26.2: type = converted %.loc22_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc22_14: type = where_expr %.Self [template = constants.%J_where.type.1] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc22_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc22: <witness> = impl_witness (constants.%empty_struct_type, <error>) [template = constants.%impl_witness.1]
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %C.ref.loc28_6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %U.ref: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %C.ref.loc28_24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc28: type = where_expr %.Self [template = constants.%J_where.type.2] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc28_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness.loc28: <witness> = impl_witness (constants.%C, <error>) [template = constants.%impl_witness.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %U: type = assoc_const_decl U [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %U [template = constants.%assoc0.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc19_29: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.loc19_29: type = converted %U.ref.loc19_29, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc19_14.1: type = splice_block %.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.1)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:       %Self.as_type.loc19_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:       %.loc19_14.2: type = converted %Self.ref, %Self.as_type.loc19_14.2 [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.1) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: <error> = value_param runtime_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [template = <error>] {
// CHECK:STDOUT:       %U.ref.loc19_23: %assoc_type = name_ref U, @J.%assoc0 [template = constants.%assoc0.1]
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
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
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
// CHECK:STDOUT: impl @impl.2: %C.ref.loc28_6 as %.loc28 {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: %C = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %C = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc33_29: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%C.ref.loc28_6 [template = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %C.ref.loc33_23: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %u: %C = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = file.%impl_witness.loc28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@J.%Self: %J.type) {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:   %Self.as_type.loc19_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc19_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc19_14.1 (%Self.as_type.1)](%u.param_patt: <error>) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%u.param_patt: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %empty_struct_type = name_ref u, %u
// CHECK:STDOUT:   return %u.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %C](%u.param_patt: %C) -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%Self.as_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.1) {
// CHECK:STDOUT:   %Self => constants.%J.facet.1
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%J.facet.2) {
// CHECK:STDOUT:   %Self => constants.%J.facet.2
// CHECK:STDOUT:   %Self.as_type.loc19_14.1 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_use_non-type_in_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [template]
// CHECK:STDOUT:   %Self.1: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %struct_type.b.1: type = struct_type {.b: %empty_struct_type} [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %M.type, %struct_type.b.1 [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @M.%Z [template]
// CHECK:STDOUT:   %G.type.1: type = fn_type @G.1 [template]
// CHECK:STDOUT:   %G.1: %G.type.1 = struct_value () [template]
// CHECK:STDOUT:   %G.assoc_type: type = assoc_entity_type %M.type, %G.type.1 [template]
// CHECK:STDOUT:   %assoc1: %G.assoc_type = assoc_entity element1, @M.%G.decl [template]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %struct_type.b.1 = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %struct: %struct_type.b.1 = struct_value (%empty_struct) [template]
// CHECK:STDOUT:   %M_where.type: type = facet_type <@M where %impl.elem0 = %struct> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct, @impl.%G.decl) [template]
// CHECK:STDOUT:   %G.type.2: type = fn_type @G.2 [template]
// CHECK:STDOUT:   %G.2: %G.type.2 = struct_value () [template]
// CHECK:STDOUT:   %M.facet: %M.type = facet_value %empty_tuple.type, %impl_witness [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .As = %import_ref.1
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
// CHECK:STDOUT:     %Z.ref: %assoc_type = name_ref Z, @M.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %struct_type.b.1 = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_33.1: %struct_type.b.1 = struct_literal (%.loc8_32)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_33.2: %empty_struct_type = converted %.loc8_32, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.b.1 = struct_value (%.loc8_33.2) [template = constants.%struct]
// CHECK:STDOUT:     %.loc8_33.3: %struct_type.b.1 = converted %.loc8_33.1, %struct [template = constants.%struct]
// CHECK:STDOUT:     %.loc8_14: type = where_expr %.Self [template = constants.%M_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc8_33.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%struct, @impl.%G.decl) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %Z: %struct_type.b.1 = assoc_const_decl Z [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %Z [template = constants.%assoc0.1]
// CHECK:STDOUT:   %G.decl: %G.type.1 = fn_decl @G.1 [template = constants.%G.1] {
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
// CHECK:STDOUT:   %G.decl: %G.type.2 = fn_decl @G.2 [template = constants.%G.2] {
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
// CHECK:STDOUT: specific @G.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G.1(constants.%M.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_associated_int_in_array.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, %i32 [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type = assoc_entity element0, @I.%N [template]
// CHECK:STDOUT:   %Self.as_type.1: type = facet_access_type %Self.1 [symbolic]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %I.type, %F.type.1 [template]
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, @I.%F.decl [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.type.3: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.19: <witness> = impl_witness (imports.%import_ref.15), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.10: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.10: %Convert.type.10 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.1, %Convert.10 [template]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %int_2.2> [template]
// CHECK:STDOUT:   %impl_witness.21: <witness> = impl_witness (%int_2.2, <error>) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2.1, bool [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %empty_tuple.type, %impl_witness.21 [template]
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
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.5
// CHECK:STDOUT:     .ImplicitAs = %import_ref.6
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
// CHECK:STDOUT:     %.loc11_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_7.2: type = converted %.loc11_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %N.ref: %assoc_type = name_ref N, @I.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc11_20: %i32 = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:     %impl.elem0.loc11_25: %Convert.type.3 = impl_witness_access constants.%impl_witness.19, element0 [template = constants.%Convert.10]
// CHECK:STDOUT:     %Convert.bound: <bound method> = bound_method %int_2, %impl.elem0.loc11_25 [template = constants.%Convert.bound]
// CHECK:STDOUT:     %Convert.specific_fn: <specific function> = specific_function %Convert.bound, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %Convert.specific_fn(%int_2) [template = constants.%int_2.2]
// CHECK:STDOUT:     %.loc11_25.1: %i32 = value_of_initializer %int.convert_checked [template = constants.%int_2.2]
// CHECK:STDOUT:     %.loc11_25.2: %i32 = converted %int_2, %.loc11_25.1 [template = constants.%int_2.2]
// CHECK:STDOUT:     %.loc11_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc11_20, %.loc11_25.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%int_2.2, <error>) [template = constants.%impl_witness.21]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %N: %i32 = assoc_const_decl N [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, %N [template = constants.%assoc0.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type.1) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %N.ref: %assoc_type = name_ref N, @I.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.loc8_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %bool.make_type, %.loc8_26.1 [template = bool]
// CHECK:STDOUT:     %.loc8_32: Core.IntLiteral = converted %N.ref, <error> [template = <error>]
// CHECK:STDOUT:     %array_type: type = array_type <error>, bool [template = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type.1) = value_param runtime_param0
// CHECK:STDOUT:     %.loc8_14.1: type = splice_block %.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.1)] {
// CHECK:STDOUT:       %Self.ref: %I.type = name_ref Self, @I.%Self [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:       %Self.as_type.loc8_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:       %.loc8_14.2: type = converted %Self.ref, %Self.as_type.loc8_14.2 [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type.1) = bind_name self, %self.param
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
// CHECK:STDOUT: impl @impl.43: %.loc11_7.2 as %.loc11_14 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %array_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %array_type = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:     %.loc12_26.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc12_26.2: type = converted %bool.make_type, %.loc12_26.1 [template = bool]
// CHECK:STDOUT:     %array_type: type = array_type %int_2, bool [template = constants.%array_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.43.%.loc11_7.2 [template = constants.%empty_tuple.type]
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.1)]
// CHECK:STDOUT:   %Self.as_type.loc8_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_14.1 (constants.%Self.as_type.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc8_14.1 (%Self.as_type.1)]() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type]() -> %return.param_patt: %array_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %.loc12_56.1: %tuple.type = tuple_literal (%true, %false)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc12_56.2: ref bool = array_index %return, %int_0
// CHECK:STDOUT:   %.loc12_56.3: init bool = initialize_from %true to %.loc12_56.2 [template = constants.%true]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %.loc12_56.4: ref bool = array_index %return, %int_1
// CHECK:STDOUT:   %.loc12_56.5: init bool = initialize_from %false to %.loc12_56.4 [template = constants.%false]
// CHECK:STDOUT:   %.loc12_56.6: init %array_type = array_init (%.loc12_56.3, %.loc12_56.5) to %return [template = constants.%array]
// CHECK:STDOUT:   %.loc12_57: init %array_type = converted %.loc12_56.1, %.loc12_56.6 [template = constants.%array]
// CHECK:STDOUT:   return %.loc12_57 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {
// CHECK:STDOUT:   %Self => constants.%Self.1
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%Self.as_type.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%I.facet) {
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc8_14.1 => constants.%empty_tuple.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
