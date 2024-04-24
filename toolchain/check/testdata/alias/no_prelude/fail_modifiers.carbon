// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+25]]:1: ERROR: `abstract` not allowed on `alias` declaration.
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+21]]:10: ERROR: `base` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+14]]:15: ERROR: `default` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:23: ERROR: `final` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:                       ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base default final alias A = Class;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `impl` not allowed on `alias` declaration.
// CHECK:STDERR: impl alias B = Class;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl alias B = Class;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `extern` not allowed on `alias` declaration.
// CHECK:STDERR: extern alias C = Class;
// CHECK:STDERR: ^~~~~~
extern alias C = Class;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.ref.loc34: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %A: type = bind_alias A, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %Class.ref.loc40: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %B: type = bind_alias B, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %Class.ref.loc45: type = name_ref Class, %Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %C: type = bind_alias C, %Class.decl [template = constants.%Class]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
