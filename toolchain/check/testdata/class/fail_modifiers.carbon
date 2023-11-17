// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+24]]:1: ERROR: Modifier not allowed.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+21]]:43: On this declaration.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR:                                           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:18: ERROR: Modifier not allowed.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR:                  ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:43: On this declaration.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR:                                           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:28: ERROR: Modifier repeated on the same declaration.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR:                            ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:9: Previously appeared here.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:37: ERROR: Modifier not allowed.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR:                                     ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:43: On this declaration.
// CHECK:STDERR: private abstract protected abstract final class ClassDefined {
// CHECK:STDERR:                                           ^
private abstract protected abstract final class ClassDefined {
}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:1: ERROR: Modifier not allowed.
// CHECK:STDERR: virtual base abstract protected class ClassDeclared;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:33: On this declaration.
// CHECK:STDERR: virtual base abstract protected class ClassDeclared;
// CHECK:STDERR:                                 ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:14: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: virtual base abstract protected class ClassDeclared;
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:9: With this modifier.
// CHECK:STDERR: virtual base abstract protected class ClassDeclared;
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:23: ERROR: Modifier not allowed.
// CHECK:STDERR: virtual base abstract protected class ClassDeclared;
// CHECK:STDERR:                       ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:33: On this declaration.
// CHECK:STDERR: virtual base abstract protected class ClassDeclared;
// CHECK:STDERR:                                 ^
virtual base abstract protected class ClassDeclared;

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc32: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_modifiers.carbon" {
// CHECK:STDOUT:   class_decl @ClassDefined, ()
// CHECK:STDOUT:   %ClassDefined: type = class_type @ClassDefined
// CHECK:STDOUT:   class_decl @ClassDeclared, ()
// CHECK:STDOUT:   %ClassDeclared: type = class_type @ClassDeclared
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ClassDefined {
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ClassDeclared;
