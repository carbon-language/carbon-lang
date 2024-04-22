// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `default` not allowed on `fn` declaration outside of an interface.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: ERROR: `protected` must appear before `default`.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `default` previously appeared here.
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default protected fn WrongOrder();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `virtual` not allowed on `fn` declaration outside of a class.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: ERROR: `virtual` repeated on declaration.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `virtual` previously appeared here.
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual virtual fn DuplicateVirtual() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: Semantics TODO: `access modifier`.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: ERROR: `protected` not allowed on declaration with `private`.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `private` previously appeared here.
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private protected fn TwoAccess();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `abstract` not allowed on `fn` declaration outside of a class.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: ERROR: `virtual` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR:          ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract virtual fn ModifiersConflict() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `base` not allowed on `fn` declaration.
// CHECK:STDERR: base fn InvalidModifier();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
base fn InvalidModifier();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:1: ERROR: `default` not allowed on `fn` declaration outside of an interface.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+14]]:9: ERROR: `final` not allowed on declaration with `default`.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: `default` previously appeared here.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:15: ERROR: `virtual` not allowed on declaration with `default`.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `default` previously appeared here.
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default final virtual fn ModifiersConflict2() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+10]]:1: ERROR: `extern` not allowed on `fn` declaration that provides a definition.
// CHECK:STDERR: extern private fn ExternOrderAndConflict() {}
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:8: ERROR: `private` must appear before `extern`.
// CHECK:STDERR: extern private fn ExternOrderAndConflict() {}
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `extern` previously appeared here.
// CHECK:STDERR: extern private fn ExternOrderAndConflict() {}
// CHECK:STDERR: ^~~~~~
extern private fn ExternOrderAndConflict() {}

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .WrongOrder = %WrongOrder
// CHECK:STDOUT:     .DuplicateVirtual = %DuplicateVirtual
// CHECK:STDOUT:     .TwoAccess = %TwoAccess
// CHECK:STDOUT:     .ModifiersConflict = %ModifiersConflict
// CHECK:STDOUT:     .InvalidModifier = %InvalidModifier
// CHECK:STDOUT:     .ModifiersConflict2 = %ModifiersConflict2
// CHECK:STDOUT:     .ExternOrderAndConflict = %ExternOrderAndConflict
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongOrder: <function> = fn_decl @WrongOrder [template] {}
// CHECK:STDOUT:   %DuplicateVirtual: <function> = fn_decl @DuplicateVirtual [template] {}
// CHECK:STDOUT:   %TwoAccess: <function> = fn_decl @TwoAccess [template] {}
// CHECK:STDOUT:   %ModifiersConflict: <function> = fn_decl @ModifiersConflict [template] {}
// CHECK:STDOUT:   %InvalidModifier: <function> = fn_decl @InvalidModifier [template] {}
// CHECK:STDOUT:   %ModifiersConflict2: <function> = fn_decl @ModifiersConflict2 [template] {}
// CHECK:STDOUT:   %ExternOrderAndConflict: <function> = fn_decl @ExternOrderAndConflict [template] {}
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
// CHECK:STDOUT: fn @ExternOrderAndConflict() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
