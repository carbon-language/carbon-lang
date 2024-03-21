// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface { }

// CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+4]]:14: ERROR: Out-of-line declaration requires a declaration in scoped entity.
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR:              ^
// CHECK:STDERR:
fn Interface.F() {}

// Nesting interfaces like this is not valid, but make sure we don't crash.
interface Outer {
  interface Inner {
    // CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+4]]:8: ERROR: Name `Outer` not found.
    // CHECK:STDERR:     fn Outer.F();
    // CHECK:STDERR:        ^~~~~
    // CHECK:STDERR:
    fn Outer.F();
  }
  // CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+3]]:12: ERROR: Out-of-line declaration requires a declaration in scoped entity.
  // CHECK:STDERR:   fn Inner.F();
  // CHECK:STDERR:            ^
  fn Inner.F();
}

// CHECK:STDOUT: --- fail_add_member_outside_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @Interface [template]
// CHECK:STDOUT:   %.2: type = interface_type @Outer [template]
// CHECK:STDOUT:   %.3: type = interface_type @Inner [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Interface = %Interface.decl
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Interface.decl: type = interface_decl @Interface [template = constants.%.1] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {}
// CHECK:STDOUT:   %Outer.decl: type = interface_decl @Outer [template = constants.%.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:   %Self: Interface = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = file.%F
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Outer {
// CHECK:STDOUT:   %Self: Outer = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %Inner.decl: type = interface_decl @Inner [template = constants.%.3] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inner {
// CHECK:STDOUT:   %Self: Inner = bind_symbolic_name Self [symbolic]
// CHECK:STDOUT:   %.loc22: <function> = fn_decl @.1 [template] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @Outer.%F
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
