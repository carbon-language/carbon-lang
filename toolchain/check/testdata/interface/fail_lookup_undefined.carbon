// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Undefined;

// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+7]]:4: ERROR: Cannot declare a member of undefined interface `Undefined`.
// CHECK:STDERR: fn Undefined.F();
// CHECK:STDERR:    ^~~~~~~~~
// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-5]]:1: Interface was forward declared here.
// CHECK:STDERR: interface Undefined;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Undefined.F();

fn Test() {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+7]]:3: ERROR: Member access into undefined interface `Undefined`.
  // CHECK:STDERR:   Undefined.G();
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-15]]:1: Interface was forward declared here.
  // CHECK:STDERR: interface Undefined;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Undefined.G();
}

interface BeingDefined {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+11]]:13: ERROR: Member access into undefined interface `BeingDefined`.
  // CHECK:STDERR:   fn H() -> BeingDefined.T;
  // CHECK:STDERR:             ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-4]]:1: Interface is currently being defined.
  // CHECK:STDERR: interface BeingDefined {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+4]]:13: ERROR: Name `T` not found.
  // CHECK:STDERR:   fn H() -> BeingDefined.T;
  // CHECK:STDERR:             ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn H() -> BeingDefined.T;
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+3]]:6: ERROR: Name `BeingDefined` not found.
  // CHECK:STDERR:   fn BeingDefined.I();
  // CHECK:STDERR:      ^~~~~~~~~~~~
  fn BeingDefined.I();
}

// CHECK:STDOUT: --- fail_lookup_undefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Undefined [template]
// CHECK:STDOUT:   %.2: type = interface_type @BeingDefined [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type @BeingDefined, <function> [template]
// CHECK:STDOUT:   %.4: <associated <function> in BeingDefined> = assoc_entity element0, @BeingDefined.%H [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Undefined = %Undefined.decl
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:     .BeingDefined = %BeingDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Undefined.decl: type = interface_decl @Undefined [template = constants.%.1] {}
// CHECK:STDOUT:   %.loc16: <function> = fn_decl @.1 [template] {}
// CHECK:STDOUT:   %Test: <function> = fn_decl @Test [template] {}
// CHECK:STDOUT:   %BeingDefined.decl: type = interface_decl @BeingDefined [template = constants.%.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Undefined;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BeingDefined {
// CHECK:STDOUT:   %Self: BeingDefined = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template] {
// CHECK:STDOUT:     %BeingDefined.ref: type = name_ref BeingDefined, file.%BeingDefined.decl [template = constants.%.2]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [template = <error>]
// CHECK:STDOUT:     %return.var: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc41: <associated <function> in BeingDefined> = assoc_entity element0, %H [template = constants.%.4]
// CHECK:STDOUT:   %.loc45: <function> = fn_decl @.2 [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %.loc41
// CHECK:STDOUT:   witness = (%H)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Undefined.ref: type = name_ref Undefined, file.%Undefined.decl [template = constants.%.1]
// CHECK:STDOUT:   %G.ref: <error> = name_ref G, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2();
// CHECK:STDOUT:
