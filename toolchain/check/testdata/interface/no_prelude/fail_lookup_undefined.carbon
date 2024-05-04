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
// CHECK:STDOUT:   %.2: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %Test: type = fn_type @Test [template]
// CHECK:STDOUT:   %struct.2: Test = struct_value () [template]
// CHECK:STDOUT:   %.4: type = interface_type @BeingDefined [template]
// CHECK:STDOUT:   %Self: BeingDefined = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.3: H = struct_value () [template]
// CHECK:STDOUT:   %.5: type = assoc_entity_type @BeingDefined, H [template]
// CHECK:STDOUT:   %.6: <associated H in BeingDefined> = assoc_entity element0, @BeingDefined.%H.decl [template]
// CHECK:STDOUT:   %.7: type = fn_type @.2 [template]
// CHECK:STDOUT:   %struct.4: <invalid> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Undefined = %Undefined.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:     .BeingDefined = %BeingDefined.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Undefined.decl: type = interface_decl @Undefined [template = constants.%.1] {}
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %Test.decl: Test = fn_decl @Test [template = constants.%struct.2] {}
// CHECK:STDOUT:   %BeingDefined.decl: type = interface_decl @BeingDefined [template = constants.%.4] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Undefined;
// CHECK:STDOUT:
// CHECK:STDOUT: interface @BeingDefined {
// CHECK:STDOUT:   %Self: BeingDefined = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.3] {
// CHECK:STDOUT:     %BeingDefined.ref: type = name_ref BeingDefined, file.%BeingDefined.decl [template = constants.%.4]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [template = <error>]
// CHECK:STDOUT:     %return.var: ref <error> = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc41: <associated H in BeingDefined> = assoc_entity element0, %H.decl [template = constants.%.6]
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.2 [template = constants.%struct.4] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .H = %.loc41
// CHECK:STDOUT:   witness = (%H.decl)
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
