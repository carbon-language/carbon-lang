// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: `default` not allowed on `fn` declaration outside of an interface.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: `protected` must appear before `default`.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `default` previously appeared here.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^~~~~~~
default protected fn WrongOrder();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: `virtual` not allowed on `fn` declaration outside of a class.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: `virtual` repeated on declaration.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `virtual` previously appeared here.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^~~~~~~
virtual virtual fn DuplicateVirtual() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: `protected` not allowed on declaration with `private`.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `private` previously appeared here.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR: ^~~~~~~
private protected fn TwoAccess();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: `abstract` not allowed on `fn` declaration outside of a class.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: `virtual` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR:          ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^~~~~~~~
abstract virtual fn ModifiersConflict() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `base` not allowed on `fn` declaration.
// CHECK:STDERR: base fn InvalidModifier();
// CHECK:STDERR: ^~~~
base fn InvalidModifier();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: ERROR: `default` not allowed on `fn` declaration outside of an interface.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:9: ERROR: `final` not allowed on declaration with `default`.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: `default` previously appeared here.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:15: ERROR: `virtual` not allowed on declaration with `default`.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `default` previously appeared here.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
default final virtual fn ModifiersConflict2() {}

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.WrongOrder = %WrongOrder, .DuplicateVirtual = %DuplicateVirtual, .TwoAccess = %TwoAccess, .ModifiersConflict = %ModifiersConflict, .InvalidModifier = %InvalidModifier, .ModifiersConflict2 = %ModifiersConflict2} [template]
// CHECK:STDOUT:   %WrongOrder: <function> = fn_decl @WrongOrder [template]
// CHECK:STDOUT:   %DuplicateVirtual: <function> = fn_decl @DuplicateVirtual [template]
// CHECK:STDOUT:   %TwoAccess: <function> = fn_decl @TwoAccess [template]
// CHECK:STDOUT:   %ModifiersConflict: <function> = fn_decl @ModifiersConflict [template]
// CHECK:STDOUT:   %InvalidModifier: <function> = fn_decl @InvalidModifier [template]
// CHECK:STDOUT:   %ModifiersConflict2: <function> = fn_decl @ModifiersConflict2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WrongOrder();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DuplicateVirtual() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TwoAccess();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ModifiersConflict() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InvalidModifier();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ModifiersConflict2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
