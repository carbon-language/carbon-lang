// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/assoc_const_in_fn_signature.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/assoc_const_in_fn_signature.carbon

// // --- core.carbon
// package Core;
//
// interface As(Dest:! type) {
//   fn Convert[self: Self]() -> Dest;
// }
//
// interface ImplicitAs(Dest:! type) {
//   fn Convert[self: Self]() -> Dest;
// }
//
// // --- fail_todo_parameter_type.carbon
// library "[[@TEST_NAME]]";
//
// import Core;

interface I {
  let T:! type;
  // CHECK:STDERR: assoc_const_in_fn_signature.carbon:[[@LINE+4]]:13: error: cannot access member of interface `I` in type `I` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   fn F() -> T;
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> T;
}

interface J {
  let U:! type;
}

interface K(C:! J) {
  fn G() -> C.U;
}

// CHECK:STDOUT: --- assoc_const_in_fn_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.c7e: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, @I.%F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self.ccd: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0.1e6: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %C: %J.type = bind_symbolic_name C, 0 [symbolic]
// CHECK:STDOUT:   %C.patt: %J.type = symbolic_binding_pattern C, 0 [symbolic]
// CHECK:STDOUT:   %K.type.7d7: type = generic_interface_type @K [concrete]
// CHECK:STDOUT:   %K.generic: %K.type.7d7 = struct_value () [concrete]
// CHECK:STDOUT:   %K.type.15b: type = facet_type <@K, @K(%C)> [symbolic]
// CHECK:STDOUT:   %Self.24f: %K.type.15b = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %C.as_type: type = facet_access_type %C [symbolic]
// CHECK:STDOUT:   %C.as_wit: <witness> = facet_access_witness %C [symbolic]
// CHECK:STDOUT:   %J.facet: %J.type = facet_value %C.as_type, %C.as_wit [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %C.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @K(%C) [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %K.type.15b [symbolic]
// CHECK:STDOUT:   %assoc0.d2b: %K.assoc_type = assoc_entity element0, @K.%G.decl [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   %K.decl: %K.type.7d7 = interface_decl @K [concrete = constants.%K.generic] {
// CHECK:STDOUT:     %C.patt.loc40_13.1: %J.type = symbolic_binding_pattern C, 0 [symbolic = %C.patt.loc40_13.2 (constants.%C.patt)]
// CHECK:STDOUT:     %C.param_patt: %J.type = value_param_pattern %C.patt.loc40_13.1, runtime_param<none> [symbolic = %C.patt.loc40_13.2 (constants.%C.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.param: %J.type = value_param runtime_param<none>
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %C.loc40_13.1: %J.type = bind_symbolic_name C, 0, %C.param [symbolic = %C.loc40_13.2 (constants.%C)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc1: %I.assoc_type = assoc_entity element1, %F.decl [concrete = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   .F = %assoc1
// CHECK:STDOUT:   witness = (%T, %F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ccd]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0.1e6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   witness = (%U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @K(%C.loc40_13.1: %J.type) {
// CHECK:STDOUT:   %C.loc40_13.2: %J.type = bind_symbolic_name C, 0 [symbolic = %C.loc40_13.2 (constants.%C)]
// CHECK:STDOUT:   %C.patt.loc40_13.2: %J.type = symbolic_binding_pattern C, 0 [symbolic = %C.patt.loc40_13.2 (constants.%C.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %K.type: type = facet_type <@K, @K(%C.loc40_13.2)> [symbolic = %K.type (constants.%K.type.15b)]
// CHECK:STDOUT:   %Self.2: %K.type.15b = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.24f)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @K(%C.loc40_13.2) [symbolic = %G.type (constants.%G.type)]
// CHECK:STDOUT:   %G: @K.%G.type (%G.type) = struct_value () [symbolic = %G (constants.%G)]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type @K.%K.type (%K.type.15b) [symbolic = %K.assoc_type (constants.%K.assoc_type)]
// CHECK:STDOUT:   %assoc0.loc41_16.2: @K.%K.assoc_type (%K.assoc_type) = assoc_entity element0, %G.decl [symbolic = %assoc0.loc41_16.2 (constants.%assoc0.d2b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @K.%K.type (%K.type.15b) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.24f)]
// CHECK:STDOUT:     %G.decl: @K.%G.type (%G.type) = fn_decl @G [symbolic = @K.%G (constants.%G)] {
// CHECK:STDOUT:       %return.patt: @G.%impl.elem0.loc41_14.1 (%impl.elem0) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @G.%impl.elem0.loc41_14.1 (%impl.elem0) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %C.ref: %J.type = name_ref C, @K.%C.loc40_13.1 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:       %U.ref: %J.assoc_type = name_ref U, @U.%assoc0 [concrete = constants.%assoc0.1e6]
// CHECK:STDOUT:       %C.as_type.loc41_14.2: type = facet_access_type %C.ref [symbolic = %C.as_type.loc41_14.1 (constants.%C.as_type)]
// CHECK:STDOUT:       %.loc41: type = converted %C.ref, %C.as_type.loc41_14.2 [symbolic = %C.as_type.loc41_14.1 (constants.%C.as_type)]
// CHECK:STDOUT:       %C.as_wit.loc41_14.2: <witness> = facet_access_witness %C.ref [symbolic = %C.as_wit.loc41_14.1 (constants.%C.as_wit)]
// CHECK:STDOUT:       %impl.elem0.loc41_14.2: type = impl_witness_access %C.as_wit.loc41_14.2, element0 [symbolic = %impl.elem0.loc41_14.1 (constants.%impl.elem0)]
// CHECK:STDOUT:       %return.param: ref @G.%impl.elem0.loc41_14.1 (%impl.elem0) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @G.%impl.elem0.loc41_14.1 (%impl.elem0) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc41_16.1: @K.%K.assoc_type (%K.assoc_type) = assoc_entity element0, %G.decl [symbolic = %assoc0.loc41_16.2 (constants.%assoc0.d2b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .G = %assoc0.loc41_16.1
// CHECK:STDOUT:     witness = (%G.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@I.%Self: %I.type) {
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@K.%C.loc40_13.1: %J.type, @K.%Self.1: @K.%K.type (%K.type.15b)) {
// CHECK:STDOUT:   %C: %J.type = bind_symbolic_name C, 0 [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.as_type.loc41_14.1: type = facet_access_type %C [symbolic = %C.as_type.loc41_14.1 (constants.%C.as_type)]
// CHECK:STDOUT:   %C.as_wit.loc41_14.1: <witness> = facet_access_witness %C [symbolic = %C.as_wit.loc41_14.1 (constants.%C.as_wit)]
// CHECK:STDOUT:   %impl.elem0.loc41_14.1: type = impl_witness_access %C.as_wit.loc41_14.1, element0 [symbolic = %impl.elem0.loc41_14.1 (constants.%impl.elem0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @G.%impl.elem0.loc41_14.1 (%impl.elem0);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self.ccd) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(constants.%C) {
// CHECK:STDOUT:   %C.loc40_13.2 => constants.%C
// CHECK:STDOUT:   %C.patt.loc40_13.2 => constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%J.facet) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%C, constants.%Self.24f) {
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %C.as_type.loc41_14.1 => constants.%C.as_type
// CHECK:STDOUT:   %C.as_wit.loc41_14.1 => constants.%C.as_wit
// CHECK:STDOUT:   %impl.elem0.loc41_14.1 => constants.%impl.elem0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @K(%C.loc40_13.2) {}
// CHECK:STDOUT:
