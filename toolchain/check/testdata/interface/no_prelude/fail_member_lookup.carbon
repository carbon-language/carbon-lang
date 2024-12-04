// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_member_lookup.carbon

interface Interface {
  fn F();

  let T:! type;
}

fn F() {
  // CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+4]]:3: error: value of type `<associated <type of F> in Interface>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   Interface.F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  Interface.F();

  // CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+4]]:10: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
  // CHECK:STDERR:   var v: Interface.T;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR:
  var v: Interface.T;
}

interface Different {}

// CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+3]]:24: error: cannot access member of interface `Interface` in type `Different` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: fn G(U:! Different) -> U.(Interface.T);
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~
fn G(U:! Different) -> U.(Interface.T);

// CHECK:STDOUT: --- fail_member_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [template]
// CHECK:STDOUT:   %Self.1: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %F.assoc_type: type = assoc_entity_type %Interface.type, %F.type.1 [template]
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, @Interface.%F.decl [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %Interface.type, type [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, @Interface.%T [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Different.type: type = facet_type <@Different> [template]
// CHECK:STDOUT:   %Self.2: %Different.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %U: %Different.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: %Different.type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Different = %Different.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %Different.decl: type = interface_decl @Different [template = constants.%Different.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %U.patt.loc36_6.1: %Different.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc36_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: %Different.type = value_param_pattern %U.patt.loc36_6.1, runtime_param<invalid> [symbolic = %U.patt.loc36_6.2 (constants.%U.patt)]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Different.ref: type = name_ref Different, file.%Different.decl [template = constants.%Different.type]
// CHECK:STDOUT:     %U.ref: %Different.type = name_ref U, %U.loc36_6.1 [symbolic = %U.loc36_6.2 (constants.%U)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, @Interface.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:     %U.param: %Different.type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc36_6.1: %Different.type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc36_6.2 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref <error> = out_param runtime_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %assoc0: %F.assoc_type = assoc_entity element0, %F.decl [template = constants.%assoc0]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, %T [template = constants.%assoc1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .T = %assoc1
// CHECK:STDOUT:   witness = (%F.decl, %T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Different {
// CHECK:STDOUT:   %Self: %Different.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Interface.ref.loc22: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   %F.ref: %F.assoc_type = name_ref F, @Interface.%assoc0 [template = constants.%assoc0]
// CHECK:STDOUT:   %Interface.ref.loc28: type = name_ref Interface, file.%Interface.decl [template = constants.%Interface.type]
// CHECK:STDOUT:   %T.ref: %assoc_type = name_ref T, @Interface.%assoc1 [template = constants.%assoc1]
// CHECK:STDOUT:   %.loc28: type = converted %T.ref, <error> [template = <error>]
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U.loc36_6.1: %Different.type) {
// CHECK:STDOUT:   %U.loc36_6.2: %Different.type = bind_symbolic_name U, 0 [symbolic = %U.loc36_6.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc36_6.2: %Different.type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc36_6.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%U.param_patt: %Different.type) -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U.loc36_6.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc36_6.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
