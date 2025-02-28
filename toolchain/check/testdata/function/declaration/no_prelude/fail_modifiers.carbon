// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:9: error: `protected` must appear before `default` [ModifierMustAppearBefore]
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default protected fn WrongOrder();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default protected fn WrongOrder();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:9: error: `virtual` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `virtual` previously appeared here [ModifierPrevious]
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `virtual` not allowed; requires class scope [ModifierRequiresClass]
// CHECK:STDERR: virtual virtual fn DuplicateVirtual() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual virtual fn DuplicateVirtual() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `protected` not allowed on declaration with `private` [ModifierNotAllowedWith]
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private protected fn TwoAccess();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private protected fn TwoAccess();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:10: error: `virtual` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR:          ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `abstract` not allowed; requires class scope [ModifierRequiresClass]
// CHECK:STDERR: abstract virtual fn ModifiersConflict() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract virtual fn ModifiersConflict() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `base` not allowed on `fn` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: base fn InvalidModifier();
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
base fn InvalidModifier();

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:9: error: `final` not allowed on declaration with `default` [ModifierNotAllowedWith]
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: note: `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:15: error: `virtual` not allowed on declaration with `default` [ModifierNotAllowedWith]
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `default` previously appeared here [ModifierPrevious]
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
// CHECK:STDERR: default final virtual fn ModifiersConflict2() {}
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default final virtual fn ModifiersConflict2() {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:8: error: `private` must appear before `extern` [ModifierMustAppearBefore]
// CHECK:STDERR: extern private fn ExternOrderAndConflict() {}
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `extern` previously appeared here [ModifierPrevious]
// CHECK:STDERR: extern private fn ExternOrderAndConflict() {}
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
extern private fn ExternOrderAndConflict() {}

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %WrongOrder.type: type = fn_type @WrongOrder [concrete]
// CHECK:STDOUT:   %WrongOrder: %WrongOrder.type = struct_value () [concrete]
// CHECK:STDOUT:   %DuplicateVirtual.type: type = fn_type @DuplicateVirtual [concrete]
// CHECK:STDOUT:   %DuplicateVirtual: %DuplicateVirtual.type = struct_value () [concrete]
// CHECK:STDOUT:   %TwoAccess.type: type = fn_type @TwoAccess [concrete]
// CHECK:STDOUT:   %TwoAccess: %TwoAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModifiersConflict.type: type = fn_type @ModifiersConflict [concrete]
// CHECK:STDOUT:   %ModifiersConflict: %ModifiersConflict.type = struct_value () [concrete]
// CHECK:STDOUT:   %InvalidModifier.type: type = fn_type @InvalidModifier [concrete]
// CHECK:STDOUT:   %InvalidModifier: %InvalidModifier.type = struct_value () [concrete]
// CHECK:STDOUT:   %ModifiersConflict2.type: type = fn_type @ModifiersConflict2 [concrete]
// CHECK:STDOUT:   %ModifiersConflict2: %ModifiersConflict2.type = struct_value () [concrete]
// CHECK:STDOUT:   %ExternOrderAndConflict.type: type = fn_type @ExternOrderAndConflict [concrete]
// CHECK:STDOUT:   %ExternOrderAndConflict: %ExternOrderAndConflict.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .WrongOrder = %WrongOrder.decl
// CHECK:STDOUT:     .DuplicateVirtual = %DuplicateVirtual.decl
// CHECK:STDOUT:     .TwoAccess [private] = %TwoAccess.decl
// CHECK:STDOUT:     .ModifiersConflict = %ModifiersConflict.decl
// CHECK:STDOUT:     .InvalidModifier = %InvalidModifier.decl
// CHECK:STDOUT:     .ModifiersConflict2 = %ModifiersConflict2.decl
// CHECK:STDOUT:     .ExternOrderAndConflict = %ExternOrderAndConflict.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongOrder.decl: %WrongOrder.type = fn_decl @WrongOrder [concrete = constants.%WrongOrder] {} {}
// CHECK:STDOUT:   %DuplicateVirtual.decl: %DuplicateVirtual.type = fn_decl @DuplicateVirtual [concrete = constants.%DuplicateVirtual] {} {}
// CHECK:STDOUT:   %TwoAccess.decl: %TwoAccess.type = fn_decl @TwoAccess [concrete = constants.%TwoAccess] {} {}
// CHECK:STDOUT:   %ModifiersConflict.decl: %ModifiersConflict.type = fn_decl @ModifiersConflict [concrete = constants.%ModifiersConflict] {} {}
// CHECK:STDOUT:   %InvalidModifier.decl: %InvalidModifier.type = fn_decl @InvalidModifier [concrete = constants.%InvalidModifier] {} {}
// CHECK:STDOUT:   %ModifiersConflict2.decl: %ModifiersConflict2.type = fn_decl @ModifiersConflict2 [concrete = constants.%ModifiersConflict2] {} {}
// CHECK:STDOUT:   %ExternOrderAndConflict.decl: %ExternOrderAndConflict.type = fn_decl @ExternOrderAndConflict [concrete = constants.%ExternOrderAndConflict] {} {}
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
