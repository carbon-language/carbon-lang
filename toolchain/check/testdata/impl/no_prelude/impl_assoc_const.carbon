// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_assoc_const.carbon

interface I { let T:! type; }

// CHECK:STDERR: impl_assoc_const.carbon:[[@LINE+4]]:1: error: semantics TODO: `impl of interface with associated constant T` [SemanticsTodo]
// CHECK:STDERR: impl () as I where .T = {} {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I where .T = {} {}

interface J {
  let U:! type;
  // CHECK:STDERR: impl_assoc_const.carbon:[[@LINE+8]]:23: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  // CHECK:STDERR: impl_assoc_const.carbon:[[@LINE+4]]:29: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   fn F[self: Self](u: U) -> U;
  // CHECK:STDERR:                             ^
  // CHECK:STDERR:
  fn F[self: Self](u: U) -> U;
}

// CHECK:STDERR: impl_assoc_const.carbon:[[@LINE+4]]:1: error: semantics TODO: `impl of interface with associated constant U` [SemanticsTodo]
// CHECK:STDERR: impl () as J where .U = {} {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as J where .U = {} {
  fn F[self: Self](u: {}) -> {} { return u; }
}

class C {}

// CHECK:STDERR: impl_assoc_const.carbon:[[@LINE+4]]:1: error: semantics TODO: `impl of interface with associated constant U` [SemanticsTodo]
// CHECK:STDERR: impl C as J where .U = C {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C as J where .U = C {
  // CHECK:STDERR: impl_assoc_const.carbon:[[@LINE+3]]:29: error: name `U` not found [NameNotFound]
  // CHECK:STDERR:   fn F[self: Self](u: C) -> U { return self; }
  // CHECK:STDERR:                             ^
  fn F[self: Self](u: C) -> U { return self; }
}

// CHECK:STDOUT: --- impl_assoc_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.1: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type.1: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0.1: %assoc_type.1 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %.Self.1: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.1: <witness> = facet_access_witness %.Self.1 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1: type = impl_witness_access %.Self.as_wit.1, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.1 = %empty_struct_type> [template]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [template]
// CHECK:STDOUT:   %Self.2: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type.2: type = assoc_entity_type %J.type, type [template]
// CHECK:STDOUT:   %assoc0.2: %assoc_type.2 = assoc_entity element0, @J.%U [template]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.2 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %J.type, %F.type.1 [template]
// CHECK:STDOUT:   %assoc1: %F.assoc_type = assoc_entity element1, @J.%F.decl [template]
// CHECK:STDOUT:   %.Self.2: %J.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %.Self.as_wit.2: <witness> = facet_access_witness %.Self.2 [symbolic]
// CHECK:STDOUT:   %impl.elem0.2: type = impl_witness_access %.Self.as_wit.2, element0 [symbolic]
// CHECK:STDOUT:   %J_where.type.1: type = facet_type <@J where %impl.elem0.2 = %empty_struct_type> [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %J_where.type.2: type = facet_type <@J where %impl.elem0.2 = %C> [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.1 [template] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self.1]
// CHECK:STDOUT:     %T.ref: %assoc_type.1 = name_ref T, @I.%assoc0 [template = constants.%assoc0.1]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit.1]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0.1]
// CHECK:STDOUT:     %.loc17_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_26.2: type = converted %.loc17_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc17_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc17_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [template = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.2 [template] {} {
// CHECK:STDOUT:     %.loc36_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc36_7.2: type = converted %.loc36_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %U.ref: %assoc_type.2 = name_ref U, @J.%assoc0 [template = constants.%assoc0.2]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit.2]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0.2]
// CHECK:STDOUT:     %.loc36_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc36_26.2: type = converted %.loc36_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc36_14: type = where_expr %.Self [template = constants.%J_where.type.1] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc36_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @impl.3 [template] {} {
// CHECK:STDOUT:     %C.ref.loc46_6: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [template = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic = constants.%.Self.2]
// CHECK:STDOUT:     %U.ref: %assoc_type.2 = name_ref U, @J.%assoc0 [template = constants.%assoc0.2]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit.2]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0.2]
// CHECK:STDOUT:     %C.ref.loc46_24: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc46: type = where_expr %.Self [template = constants.%J_where.type.2] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %C.ref.loc46_24
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.1 = assoc_entity element0, %T [template = constants.%assoc0.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:   %U: type = assoc_const_decl U [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.2 = assoc_entity element0, %U [template = constants.%assoc0.2]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: @F.1.%Self.as_type.loc29_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.1.%Self.as_type.loc29_14.1 (%Self.as_type) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: <error> = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: <error> = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc29_29: %assoc_type.2 = name_ref U, @J.%assoc0 [template = constants.%assoc0.2]
// CHECK:STDOUT:     %.loc29_29: type = converted %U.ref.loc29_29, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: @F.1.%Self.as_type.loc29_14.1 (%Self.as_type) = value_param runtime_param0
// CHECK:STDOUT:     %.loc29_14.1: type = splice_block %.loc29_14.2 [symbolic = %Self.as_type.loc29_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:       %Self.ref: %J.type = name_ref Self, @J.%Self [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:       %Self.as_type.loc29_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc29_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       %.loc29_14.2: type = converted %Self.ref, %Self.as_type.loc29_14.2 [symbolic = %Self.as_type.loc29_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: @F.1.%Self.as_type.loc29_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: <error> = value_param runtime_param1
// CHECK:STDOUT:     %.1: <error> = splice_block <error> [template = <error>] {
// CHECK:STDOUT:       %U.ref.loc29_23: %assoc_type.2 = name_ref U, @J.%assoc0 [template = constants.%assoc0.2]
// CHECK:STDOUT:       %.loc29_23: type = converted %U.ref.loc29_23, <error> [template = <error>]
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
// CHECK:STDOUT: impl @impl.1: %.loc17_7.2 as %.loc17_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.2: %.loc36_7.2 as %.loc36_14 {
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %empty_tuple.type = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %empty_tuple.type = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: %empty_struct_type = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %empty_struct_type = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc37_31.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc37_31.2: type = converted %.loc37_31.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %self.param: %empty_tuple.type = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.2.%.loc36_7.2 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %self: %empty_tuple.type = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %empty_struct_type = value_param runtime_param1
// CHECK:STDOUT:     %.loc37_24.1: type = splice_block %.loc37_24.3 [template = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc37_24.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:       %.loc37_24.3: type = converted %.loc37_24.2, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %u: %empty_struct_type = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param runtime_param2
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3: %C.ref.loc46_6 as %.loc46 {
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %u.patt: %C = binding_pattern u
// CHECK:STDOUT:     %u.param_patt: %C = value_param_pattern %u.patt, runtime_param1
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param2
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: %C = value_param runtime_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.3.%C.ref.loc46_6 [template = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %u.param: %C = value_param runtime_param1
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %u: %C = bind_name u, %u.param
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param2
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
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
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = %Self (constants.%Self.2)]
// CHECK:STDOUT:   %Self.as_type.loc29_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc29_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc29_14.1 (%Self.as_type)](%u.param_patt: <error>) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self.param_patt: %empty_tuple.type](%u.param_patt: %empty_struct_type) -> %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %u.ref: %empty_struct_type = name_ref u, %u
// CHECK:STDOUT:   return %u.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[%self.param_patt: %C](%u.param_patt: %C) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %C = name_ref self, %self
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.2) {
// CHECK:STDOUT:   %Self => constants.%Self.2
// CHECK:STDOUT:   %Self.as_type.loc29_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
