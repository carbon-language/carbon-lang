// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface Interface { }

// CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+3]]:14: ERROR: Out-of-line declaration requires a declaration in scoped entity.
// CHECK:STDERR: fn Interface.F() {}
// CHECK:STDERR:              ^
fn Interface.F() {}

// Nesting interfaces like this is not valid, but make sure we don't crash.
interface Outer {
  interface Inner {
    // CHECK:STDERR: fail_add_member_outside_definition.carbon:[[@LINE+3]]:8: ERROR: Name `Outer` not found.
    // CHECK:STDERR:     fn Outer.F();
    // CHECK:STDERR:        ^~~~~
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
// CHECK:STDOUT:   %.2: type = assoc_entity_type @Interface, <function> [template]
// CHECK:STDOUT:   %.3: <associated <function> in Interface> = assoc_entity element0, file.%F [template]
// CHECK:STDOUT:   %.4: type = interface_type @Outer [template]
// CHECK:STDOUT:   %.5: type = interface_type @Inner [template]
// CHECK:STDOUT:   %.6: type = assoc_entity_type @Inner, <function> [template]
// CHECK:STDOUT:   %.7: <associated <function> in Inner> = assoc_entity element0, @Outer.%F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Interface = %Interface.decl, .Outer = %Outer.decl} [template]
// CHECK:STDOUT:   %Interface.decl = interface_decl @Interface, () [template = constants.%.1]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template]
// CHECK:STDOUT:   %.loc12: <associated <function> in Interface> = assoc_entity element0, %F [template = constants.%.3]
// CHECK:STDOUT:   %Outer.decl = interface_decl @Outer, () [template = constants.%.4]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Interface {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = file.%.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Outer {
// CHECK:STDOUT:   %Inner.decl = interface_decl @Inner, () [template = constants.%.5]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template]
// CHECK:STDOUT:   %.loc25: <associated <function> in Inner> = assoc_entity element0, %F [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Inner = %Inner.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Inner {
// CHECK:STDOUT:   %.loc20: <function> = fn_decl @.1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = @Outer.%.loc25
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
