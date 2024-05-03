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
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %WrongOrder: type = fn_type @WrongOrder [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: WrongOrder = struct_value () [template]
// CHECK:STDOUT:   %DuplicateVirtual: type = fn_type @DuplicateVirtual [template]
// CHECK:STDOUT:   %struct.2: DuplicateVirtual = struct_value () [template]
// CHECK:STDOUT:   %TwoAccess: type = fn_type @TwoAccess [template]
// CHECK:STDOUT:   %struct.3: TwoAccess = struct_value () [template]
// CHECK:STDOUT:   %ModifiersConflict: type = fn_type @ModifiersConflict [template]
// CHECK:STDOUT:   %struct.4: ModifiersConflict = struct_value () [template]
// CHECK:STDOUT:   %InvalidModifier: type = fn_type @InvalidModifier [template]
// CHECK:STDOUT:   %struct.5: InvalidModifier = struct_value () [template]
// CHECK:STDOUT:   %ModifiersConflict2: type = fn_type @ModifiersConflict2 [template]
// CHECK:STDOUT:   %struct.6: ModifiersConflict2 = struct_value () [template]
// CHECK:STDOUT:   %ExternOrderAndConflict: type = fn_type @ExternOrderAndConflict [template]
// CHECK:STDOUT:   %struct.7: ExternOrderAndConflict = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .WrongOrder = %WrongOrder.decl
// CHECK:STDOUT:     .DuplicateVirtual = %DuplicateVirtual.decl
// CHECK:STDOUT:     .TwoAccess = %TwoAccess.decl
// CHECK:STDOUT:     .ModifiersConflict = %ModifiersConflict.decl
// CHECK:STDOUT:     .InvalidModifier = %InvalidModifier.decl
// CHECK:STDOUT:     .ModifiersConflict2 = %ModifiersConflict2.decl
// CHECK:STDOUT:     .ExternOrderAndConflict = %ExternOrderAndConflict.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongOrder.decl: WrongOrder = fn_decl @WrongOrder [template = constants.%struct.1] {}
// CHECK:STDOUT:   %DuplicateVirtual.decl: DuplicateVirtual = fn_decl @DuplicateVirtual [template = constants.%struct.2] {}
// CHECK:STDOUT:   %TwoAccess.decl: TwoAccess = fn_decl @TwoAccess [template = constants.%struct.3] {}
// CHECK:STDOUT:   %ModifiersConflict.decl: ModifiersConflict = fn_decl @ModifiersConflict [template = constants.%struct.4] {}
// CHECK:STDOUT:   %InvalidModifier.decl: InvalidModifier = fn_decl @InvalidModifier [template = constants.%struct.5] {}
// CHECK:STDOUT:   %ModifiersConflict2.decl: ModifiersConflict2 = fn_decl @ModifiersConflict2 [template = constants.%struct.6] {}
// CHECK:STDOUT:   %ExternOrderAndConflict.decl: ExternOrderAndConflict = fn_decl @ExternOrderAndConflict [template = constants.%struct.7] {}
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
