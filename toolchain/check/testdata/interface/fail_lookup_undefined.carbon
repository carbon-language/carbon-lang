// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Undefined;

// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+6]]:4: ERROR: Cannot declare a member of undefined interface `Undefined`.
// CHECK:STDERR: fn Undefined.F();
// CHECK:STDERR:    ^~~~~~~~~
// CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-5]]:1: Interface was forward declared here.
// CHECK:STDERR: interface Undefined;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
fn Undefined.F();

fn Test() {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+6]]:3: ERROR: Member access into undefined interface `Undefined`.
  // CHECK:STDERR:   Undefined.G();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-14]]:1: Interface was forward declared here.
  // CHECK:STDERR: interface Undefined;
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  Undefined.G();
}

interface BeingDefined {
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+9]]:13: ERROR: Member access into undefined interface `BeingDefined`.
  // CHECK:STDERR:   fn H() -> BeingDefined.T;
  // CHECK:STDERR:             ^~~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE-4]]:1: Interface is currently being defined.
  // CHECK:STDERR: interface BeingDefined {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_lookup_undefined.carbon:[[@LINE+3]]:13: ERROR: Name `T` not found.
  // CHECK:STDERR:   fn H() -> BeingDefined.T;
  // CHECK:STDERR:             ^~~~~~~~~~~~~~
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
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .Undefined = %Undefined.decl
// CHECK:STDOUT:     .Test = %Test
// CHECK:STDOUT:     .BeingDefined = %BeingDefined.decl
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %Undefined.decl = interface_decl @Undefined, () [template = constants.%.1]
// CHECK:STDOUT:   %.loc15: <function> = fn_decl @.1 [template]
// CHECK:STDOUT:   %Test: <function> = fn_decl @Test [template]
// CHECK:STDOUT:   %BeingDefined.decl = interface_decl @BeingDefined, () [template = constants.%.2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Undefined;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BeingDefined {
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template]
// CHECK:STDOUT:   %.loc41: <function> = fn_decl @.2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .H = %H
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
