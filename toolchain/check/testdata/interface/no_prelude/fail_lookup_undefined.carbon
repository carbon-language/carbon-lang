// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_lookup_undefined.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_lookup_undefined.carbon

interface Undefined;

// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+7]]:4: error: cannot declare a member of undefined interface `Undefined`
// CHECK:STDERR: fn Undefined.F();
// CHECK:STDERR:    ^~~~~~~~~
// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-5]]:1: note: interface was forward declared here
// CHECK:STDERR: interface Undefined;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Undefined.F();

fn Test() {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+7]]:3: error: member access into undefined interface `Undefined`
  // CHECK:STDERR:   Undefined.G();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-15]]:1: note: interface was forward declared here
  // CHECK:STDERR: interface Undefined;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Undefined.G();
}

interface BeingDefined {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+11]]:13: error: member access into undefined interface `BeingDefined`
  // CHECK:STDERR:   fn H() -> BeingDefined.T;
  // CHECK:STDERR:             ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-4]]:1: note: interface is currently being defined
  // CHECK:STDERR: interface BeingDefined {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+4]]:13: error: name `T` not found
  // CHECK:STDERR:   fn H() -> BeingDefined.T;
  // CHECK:STDERR:             ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn H() -> BeingDefined.T;
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+3]]:6: error: name `BeingDefined` not found
  // CHECK:STDERR:   fn BeingDefined.I();
  // CHECK:STDERR:      ^~~~~~~~~~~~
  fn BeingDefined.I();
}

// CHECK:STDOUT: --- fail_lookup_undefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Undefined.type: type = interface_type @Undefined [template]
// CHECK:STDOUT:   %.type.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: %.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [template]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [template]
// CHECK:STDOUT:   %BeingDefined.type: type = interface_type @BeingDefined [template]
// CHECK:STDOUT:   %Self: %BeingDefined.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %BeingDefined.type, %H.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @BeingDefined.%H.decl [template]
// CHECK:STDOUT:   %.type.2: type = fn_type @.2 [template]
// CHECK:STDOUT:   %.5: %.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Undefined = %Undefined.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .BeingDefined = %BeingDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Undefined.decl: type = interface_decl @Undefined [template = constants.%Undefined.type] {} {}
// CHECK:STDOUT:   %.decl: %.type.1 = fn_decl @.1 [template = constants.%.2] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [template = constants.%Test] {} {}
// CHECK:STDOUT:   %BeingDefined.decl: type = interface_decl @BeingDefined [template = constants.%BeingDefined.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Undefined;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BeingDefined {
// CHECK:STDOUT:   %Self: %BeingDefined.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {
// CHECK:STDOUT:     %BeingDefined.ref: type = name_ref BeingDefined, file.%BeingDefined.decl [template = constants.%BeingDefined.type]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [template = <error>]
// CHECK:STDOUT:     %return: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc45: %.3 = assoc_entity element0, %H.decl [template = constants.%.4]
// CHECK:STDOUT:   %.decl: %.type.2 = fn_decl @.2 [template = constants.%.5] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %.loc45
// CHECK:STDOUT:   witness = (%H.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Undefined.ref: type = name_ref Undefined, file.%Undefined.decl [template = constants.%Undefined.type]
// CHECK:STDOUT:   %G.ref: <error> = name_ref G, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(@BeingDefined.%Self: %BeingDefined.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> <error>;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.2(@BeingDefined.%Self: %BeingDefined.type) {
// CHECK:STDOUT:
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.2(constants.%Self) {}
// CHECK:STDOUT:
