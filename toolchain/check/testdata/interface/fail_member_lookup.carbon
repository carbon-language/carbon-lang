// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_member_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_member_lookup.carbon

// --- fail_member_lookup.carbon
library "[[@TEST_NAME]]";

interface Interface {
  fn F();

  let T:! type;
}

fn F() {
  // CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+4]]:3: error: value of type `<associated entity in Interface>` is not callable [CallToNonCallable]
  // CHECK:STDERR:   Interface.F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  Interface.F();

  // CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+7]]:10: error: cannot implicitly convert non-type value of type `<associated entity in Interface>` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   var v: Interface.T;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+4]]:10: note: type `<associated entity in Interface>` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   var v: Interface.T;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR:
  var v: Interface.T;
}

interface Different {}

// CHECK:STDERR: fail_member_lookup.carbon:[[@LINE+4]]:24: error: cannot convert type `U` that implements `Different` into type implementing `Interface` [ConversionFailureFacetToFacet]
// CHECK:STDERR: fn G(U:! Different) -> U.(Interface.T);
// CHECK:STDERR:                        ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(U:! Different) -> U.(Interface.T);

// CHECK:STDOUT: --- fail_member_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Interface.type: type = facet_type <@Interface> [concrete]
// CHECK:STDOUT:   %Self.46e: %Interface.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %Interface.F.type: type = fn_type @Interface.F [concrete]
// CHECK:STDOUT:   %Interface.F: %Interface.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Interface.assoc_type: type = assoc_entity_type @Interface [concrete]
// CHECK:STDOUT:   %assoc0.b4c: %Interface.assoc_type = assoc_entity element0, @Interface.%Interface.F.decl [concrete]
// CHECK:STDOUT:   %assoc1: %Interface.assoc_type = assoc_entity element1, @Interface.%T [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Different.type: type = facet_type <@Different> [concrete]
// CHECK:STDOUT:   %Self.b8f: %Different.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U: %Different.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.fa7: type = pattern_type %Different.type [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .Different = %Different.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [concrete = constants.%Interface.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %Different.decl: type = interface_decl @Different [concrete = constants.%Different.type] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt: %pattern_type.fa7 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %return.patt: <error> = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: <error> = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: %Different.type = name_ref U, %U.loc32_6.2 [symbolic = %U.loc32_6.1 (constants.%U)]
// CHECK:STDOUT:     %Interface.ref: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %T.ref: %Interface.assoc_type = name_ref T, @T.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.loc32: type = splice_block %Different.ref [concrete = constants.%Different.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Different.ref: type = name_ref Different, file.%Different.decl [concrete = constants.%Different.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc32_6.2: %Different.type = bind_symbolic_name U, 0 [symbolic = %U.loc32_6.1 (constants.%U)]
// CHECK:STDOUT:     %return.param: ref <error> = out_param call_param0
// CHECK:STDOUT:     %return: ref <error> = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: %Interface.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.46e]
// CHECK:STDOUT:   %Interface.F.decl: %Interface.F.type = fn_decl @Interface.F [concrete = constants.%Interface.F] {} {}
// CHECK:STDOUT:   %assoc0: %Interface.assoc_type = assoc_entity element0, %Interface.F.decl [concrete = constants.%assoc0.b4c]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc1: %Interface.assoc_type = assoc_entity element1, @Interface.%T [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   .T = @T.%assoc1
// CHECK:STDOUT:   witness = (%Interface.F.decl, %T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Different {
// CHECK:STDOUT:   %Self: %Different.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.b8f]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Interface.F(@Interface.%Self: %Interface.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Interface.ref.loc14: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:   %F.ref: %Interface.assoc_type = name_ref F, @Interface.%assoc0 [concrete = constants.%assoc0.b4c]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: <error> = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var %v.var_patt [concrete = <error>]
// CHECK:STDOUT:   %.1: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:     %Interface.ref.loc23: type = name_ref Interface, file.%Interface.decl [concrete = constants.%Interface.type]
// CHECK:STDOUT:     %T.ref: %Interface.assoc_type = name_ref T, @T.%assoc1 [concrete = constants.%assoc1]
// CHECK:STDOUT:     %.loc23: type = converted %T.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: <error> = bind_name v, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U.loc32_6.2: %Different.type) {
// CHECK:STDOUT:   %U.loc32_6.1: %Different.type = bind_symbolic_name U, 0 [symbolic = %U.loc32_6.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Interface.F(constants.%Self.46e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.46e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U.loc32_6.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
