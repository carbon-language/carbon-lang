// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_lookup_undefined.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_lookup_undefined.carbon

interface Undefined;

// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+7]]:4: error: cannot declare a member of undefined interface `Undefined` [QualifiedDeclInUndefinedInterfaceScope]
// CHECK:STDERR: fn Undefined.F();
// CHECK:STDERR:    ^~~~~~~~~
// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface Undefined;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Undefined.F();

fn Test() {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+7]]:3: error: member access into incomplete facet type `Undefined` [QualifiedExprInIncompleteFacetTypeScope]
  // CHECK:STDERR:   Undefined.G();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-15]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface Undefined;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Undefined.G();
}

interface BeingDefined {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+7]]:13: error: member access into incomplete facet type `BeingDefined` [QualifiedExprInIncompleteFacetTypeScope]
  // CHECK:STDERR:   fn H() -> BeingDefined.T;
  // CHECK:STDERR:             ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-4]]:1: note: interface is currently being defined [InterfaceIncompleteWithinDefinition]
  // CHECK:STDERR: interface BeingDefined {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn H() -> BeingDefined.T;
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+4]]:6: error: name `BeingDefined` not found [NameNotFound]
  // CHECK:STDERR:   fn BeingDefined.I();
  // CHECK:STDERR:      ^~~~~~~~~~~~
  // CHECK:STDERR:
  fn BeingDefined.I();
}

// CHECK:STDOUT: --- fail_lookup_undefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Undefined.type: type = facet_type <@Undefined> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %BeingDefined.type: type = facet_type <@BeingDefined> [concrete]
// CHECK:STDOUT:   %Self: %BeingDefined.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %BeingDefined+Self.H.type: type = fn_type @BeingDefined+Self.H, @BeingDefined(%Self) [symbolic]
// CHECK:STDOUT:   %BeingDefined+Self.H: %BeingDefined+Self.H.type = struct_value () [symbolic]
// CHECK:STDOUT:   %BeingDefined.assoc_type: type = assoc_entity_type @BeingDefined [concrete]
// CHECK:STDOUT:   %assoc0: %BeingDefined.assoc_type = assoc_entity element0, @BeingDefined.%BeingDefined+Self.H.decl [concrete]
// CHECK:STDOUT:   %BeingDefined+Self.I.type: type = fn_type @BeingDefined+Self.I, @BeingDefined(%Self) [symbolic]
// CHECK:STDOUT:   %BeingDefined+Self.I: %BeingDefined+Self.I.type = struct_value () [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Undefined = %Undefined.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .BeingDefined = %BeingDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Undefined.decl: type = interface_decl @Undefined [concrete = constants.%Undefined.type] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {} {}
// CHECK:STDOUT:   %BeingDefined.decl: type = interface_decl @BeingDefined [concrete = constants.%BeingDefined.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Undefined;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BeingDefined {
// CHECK:STDOUT:   %Self.loc37: %BeingDefined.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   interface_with_self_decl @BeingDefined [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %BeingDefined+Self.H.decl: @BeingDefined.%BeingDefined+Self.H.type (%BeingDefined+Self.H.type) = fn_decl @BeingDefined+Self.H [symbolic = @BeingDefined.%BeingDefined+Self.H (constants.%BeingDefined+Self.H)] {} {
// CHECK:STDOUT:     %BeingDefined.ref: type = name_ref BeingDefined, file.%BeingDefined.decl [concrete = constants.%BeingDefined.type]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BeingDefined.assoc_type = assoc_entity element0, %BeingDefined+Self.H.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %BeingDefined+Self.I.decl: @BeingDefined.%BeingDefined+Self.I.type (%BeingDefined+Self.I.type) = fn_decl @BeingDefined+Self.I [symbolic = @BeingDefined.%BeingDefined+Self.I (constants.%BeingDefined+Self.I)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self.loc37
// CHECK:STDOUT:   .BeingDefined = <poisoned>
// CHECK:STDOUT:   .BeingDefined = <poisoned>
// CHECK:STDOUT:   .H = %assoc0
// CHECK:STDOUT:   witness = (%BeingDefined+Self.H.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Undefined.ref: type = name_ref Undefined, file.%Undefined.decl [concrete = constants.%Undefined.type]
// CHECK:STDOUT:   %G.ref: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BeingDefined+Self.H(@BeingDefined.%Self.loc37: %BeingDefined.type) {
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BeingDefined+Self.I(@BeingDefined.%Self.loc37: %BeingDefined.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BeingDefined(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BeingDefined+Self.H(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BeingDefined+Self.I(constants.%Self) {}
// CHECK:STDOUT:
