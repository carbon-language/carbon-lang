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
// CHECK:STDOUT:   %BeingDefined.WithSelf.H.type: type = fn_type @BeingDefined.WithSelf.H, @BeingDefined.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %BeingDefined.WithSelf.H: %BeingDefined.WithSelf.H.type = struct_value () [symbolic]
// CHECK:STDOUT:   %BeingDefined.assoc_type: type = assoc_entity_type @BeingDefined [concrete]
// CHECK:STDOUT:   %assoc0: %BeingDefined.assoc_type = assoc_entity element0, @BeingDefined.WithSelf.%BeingDefined.WithSelf.H.decl [concrete]
// CHECK:STDOUT:   %BeingDefined.WithSelf.I.type: type = fn_type @BeingDefined.WithSelf.I, @BeingDefined.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %BeingDefined.WithSelf.I: %BeingDefined.WithSelf.I.type = struct_value () [symbolic]
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
// CHECK:STDOUT:   %Self: %BeingDefined.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %BeingDefined.WithSelf.decl = interface_with_self_decl @BeingDefined [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %BeingDefined.WithSelf.H.decl: @BeingDefined.WithSelf.%BeingDefined.WithSelf.H.type (%BeingDefined.WithSelf.H.type) = fn_decl @BeingDefined.WithSelf.H [symbolic = @BeingDefined.WithSelf.%BeingDefined.WithSelf.H (constants.%BeingDefined.WithSelf.H)] {} {
// CHECK:STDOUT:     %BeingDefined.ref: type = name_ref BeingDefined, file.%BeingDefined.decl [concrete = constants.%BeingDefined.type]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %assoc0: %BeingDefined.assoc_type = assoc_entity element0, %BeingDefined.WithSelf.H.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:   %BeingDefined.WithSelf.I.decl: @BeingDefined.WithSelf.%BeingDefined.WithSelf.I.type (%BeingDefined.WithSelf.I.type) = fn_decl @BeingDefined.WithSelf.I [symbolic = @BeingDefined.WithSelf.%BeingDefined.WithSelf.I (constants.%BeingDefined.WithSelf.I)] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .BeingDefined = <poisoned>
// CHECK:STDOUT:   .BeingDefined = <poisoned>
// CHECK:STDOUT:   .H = @BeingDefined.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@BeingDefined.WithSelf.%BeingDefined.WithSelf.H.decl)
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
// CHECK:STDOUT: generic fn @BeingDefined.WithSelf.H(@BeingDefined.%Self: %BeingDefined.type) {
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @BeingDefined.WithSelf.I(@BeingDefined.%Self: %BeingDefined.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BeingDefined.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BeingDefined.WithSelf.H(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @BeingDefined.WithSelf.I(constants.%Self) {}
// CHECK:STDOUT:
