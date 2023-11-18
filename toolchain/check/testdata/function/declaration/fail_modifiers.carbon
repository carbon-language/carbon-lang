// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `default`.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: Modifier must appear earlier.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Before this modifier.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^
default protected fn WrongOrder();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `virtual`.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: Modifier repeated on the same declaration.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Previously appeared here.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^
virtual virtual fn DuplicateVirtual() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `private`.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR: ^
private protected fn TwoAccess();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `abstract`.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:10: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^
abstract virtual fn ModifiersConflict() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:1: ERROR: Modifier not allowed.
// CHECK:STDERR: base fn InvalidModifier();
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:6: On this declaration.
// CHECK:STDERR: base fn InvalidModifier();
// CHECK:STDERR:      ^
base fn InvalidModifier();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: ERROR: Semantics TODO: `default`.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:9: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: With this modifier.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:15: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:               ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^
default final virtual fn ModifiersConflict2() {}

// CHECK:STDOUT: file "fail_modifiers.carbon" {
// CHECK:STDOUT:   %WrongOrder: <function> = fn_decl @WrongOrder
// CHECK:STDOUT:   %DuplicateVirtual: <function> = fn_decl @DuplicateVirtual
// CHECK:STDOUT:   %TwoAccess: <function> = fn_decl @TwoAccess
// CHECK:STDOUT:   %ModifiersConflict: <function> = fn_decl @ModifiersConflict
// CHECK:STDOUT:   %InvalidModifier: <function> = fn_decl @InvalidModifier
// CHECK:STDOUT:   %ModifiersConflict2: <function> = fn_decl @ModifiersConflict2
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
