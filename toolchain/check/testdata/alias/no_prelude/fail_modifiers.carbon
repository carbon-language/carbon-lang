// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_modifiers.carbon

class Class {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+25]]:10: error: `base` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+22]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:15: error: `default` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:23: error: `final` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR:                       ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `alias` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: abstract base default final alias A = Class;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base default final alias A = Class;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `impl` not allowed on `alias` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: impl alias B = Class;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl alias B = Class;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `extern` not allowed on `alias` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: extern alias C = Class;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
extern alias C = Class;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Class.ref.loc38: type = name_ref Class, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %A: type = bind_alias A, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %Class.ref.loc44: type = name_ref Class, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %B: type = bind_alias B, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %Class.ref.loc50: type = name_ref Class, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %C: type = bind_alias C, %Class.decl [concrete = constants.%Class]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
