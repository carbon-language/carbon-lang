// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_modifiers.carbon

base class B {}

class C1 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: error: `private` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   private extend base: B;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private extend base: B;
}

class C2 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+8]]:3: error: `abstract` not allowed on `base` declaration [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: error: missing `extend` before `base` declaration [BaseMissingExtend]
  // CHECK:STDERR:   abstract base: B;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract base: B;
}

class C3 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+7]]:10: error: `default` not allowed on declaration with `extend` [ModifierNotAllowedWith]
  // CHECK:STDERR:   extend default base: B;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: note: `extend` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:   extend default base: B;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend default base: B;
}

class C4 {
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+7]]:10: error: `extend` repeated on declaration [ModifierRepeated]
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_base_modifiers.carbon:[[@LINE+4]]:3: note: `extend` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:   extend extend base: B;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend extend base: B;
}

// CHECK:STDOUT: --- fail_base_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %C1.elem: type = unbound_element_type %C1, %B [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %C2.elem: type = unbound_element_type %C2, %B [concrete]
// CHECK:STDOUT:   %C3: type = class_type @C3 [concrete]
// CHECK:STDOUT:   %C3.elem: type = unbound_element_type %C3, %B [concrete]
// CHECK:STDOUT:   %C4: type = class_type @C4 [concrete]
// CHECK:STDOUT:   %C4.elem: type = unbound_element_type %C4, %B [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:     .C3 = %C3.decl
// CHECK:STDOUT:     .C4 = %C4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [concrete = constants.%C1] {} {}
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   %C3.decl: type = class_decl @C3 [concrete = constants.%C3] {} {}
// CHECK:STDOUT:   %C4.decl: type = class_decl @C4 [concrete = constants.%C4] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc18: %C1.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc30: %C2.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc41: %C3.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C3
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc41
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc52: %C4.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C4
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc52
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
