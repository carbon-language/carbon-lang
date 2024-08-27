// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_modifiers.carbon

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
// CHECK:STDOUT:   %WrongOrder.type: type = fn_type @WrongOrder [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %WrongOrder: %WrongOrder.type = struct_value () [template]
// CHECK:STDOUT:   %DuplicateVirtual.type: type = fn_type @DuplicateVirtual [template]
// CHECK:STDOUT:   %DuplicateVirtual: %DuplicateVirtual.type = struct_value () [template]
// CHECK:STDOUT:   %TwoAccess.type: type = fn_type @TwoAccess [template]
// CHECK:STDOUT:   %TwoAccess: %TwoAccess.type = struct_value () [template]
// CHECK:STDOUT:   %ModifiersConflict.type: type = fn_type @ModifiersConflict [template]
// CHECK:STDOUT:   %ModifiersConflict: %ModifiersConflict.type = struct_value () [template]
// CHECK:STDOUT:   %InvalidModifier.type: type = fn_type @InvalidModifier [template]
// CHECK:STDOUT:   %InvalidModifier: %InvalidModifier.type = struct_value () [template]
// CHECK:STDOUT:   %ModifiersConflict2.type: type = fn_type @ModifiersConflict2 [template]
// CHECK:STDOUT:   %ModifiersConflict2: %ModifiersConflict2.type = struct_value () [template]
// CHECK:STDOUT:   %ExternOrderAndConflict.type: type = fn_type @ExternOrderAndConflict [template]
// CHECK:STDOUT:   %ExternOrderAndConflict: %ExternOrderAndConflict.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .WrongOrder = %WrongOrder.decl
// CHECK:STDOUT:     .DuplicateVirtual = %DuplicateVirtual.decl
// CHECK:STDOUT:     .TwoAccess [private] = %TwoAccess.decl
// CHECK:STDOUT:     .ModifiersConflict = %ModifiersConflict.decl
// CHECK:STDOUT:     .InvalidModifier = %InvalidModifier.decl
// CHECK:STDOUT:     .ModifiersConflict2 = %ModifiersConflict2.decl
// CHECK:STDOUT:     .ExternOrderAndConflict = %ExternOrderAndConflict.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongOrder.decl: %WrongOrder.type = fn_decl @WrongOrder [template = constants.%WrongOrder] {}
// CHECK:STDOUT:   %DuplicateVirtual.decl: %DuplicateVirtual.type = fn_decl @DuplicateVirtual [template = constants.%DuplicateVirtual] {}
// CHECK:STDOUT:   %TwoAccess.decl: %TwoAccess.type = fn_decl @TwoAccess [template = constants.%TwoAccess] {}
// CHECK:STDOUT:   %ModifiersConflict.decl: %ModifiersConflict.type = fn_decl @ModifiersConflict [template = constants.%ModifiersConflict] {}
// CHECK:STDOUT:   %InvalidModifier.decl: %InvalidModifier.type = fn_decl @InvalidModifier [template = constants.%InvalidModifier] {}
// CHECK:STDOUT:   %ModifiersConflict2.decl: %ModifiersConflict2.type = fn_decl @ModifiersConflict2 [template = constants.%ModifiersConflict2] {}
// CHECK:STDOUT:   %ExternOrderAndConflict.decl: %ExternOrderAndConflict.type = fn_decl @ExternOrderAndConflict [template = constants.%ExternOrderAndConflict] {}
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
// CHECK:STDOUT: extern fn @ExternOrderAndConflict() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
