// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class B {}

class C1 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+4]]:3: ERROR: `private` not allowed on `adapt` declaration.
  // CHECK:STDERR:   private adapt B;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private adapt B;
}

class C2 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+4]]:3: ERROR: `abstract` not allowed on `adapt` declaration.
  // CHECK:STDERR:   abstract adapt B;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  abstract adapt B;
}

class C3 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+4]]:3: ERROR: `default` not allowed on `adapt` declaration.
  // CHECK:STDERR:   default adapt B;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default adapt B;
}

class C4 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+7]]:10: ERROR: `extend` repeated on declaration.
  // CHECK:STDERR:   extend extend adapt B;
  // CHECK:STDERR:          ^~~~~~
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+4]]:3: `extend` previously appeared here.
  // CHECK:STDERR:   extend extend adapt B;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extend extend adapt B;
}

class C5 {
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+6]]:10: ERROR: `base` not allowed on declaration with `extend`.
  // CHECK:STDERR:   extend base adapt B;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_adapt_modifiers.carbon:[[@LINE+3]]:3: `extend` previously appeared here.
  // CHECK:STDERR:   extend base adapt B;
  // CHECK:STDERR:   ^~~~~~
  extend base adapt B;
}

// CHECK:STDOUT: --- fail_adapt_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %C1: type = class_type @C1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %C2: type = class_type @C2 [template]
// CHECK:STDOUT:   %C3: type = class_type @C3 [template]
// CHECK:STDOUT:   %C4: type = class_type @C4 [template]
// CHECK:STDOUT:   %C5: type = class_type @C5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:     .C3 = %C3.decl
// CHECK:STDOUT:     .C4 = %C4.decl
// CHECK:STDOUT:     .C5 = %C5.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [template = constants.%C1] {}
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [template = constants.%C2] {}
// CHECK:STDOUT:   %C3.decl: type = class_decl @C3 [template = constants.%C3] {}
// CHECK:STDOUT:   %C4.decl: type = class_decl @C4 [template = constants.%C4] {}
// CHECK:STDOUT:   %C5.decl: type = class_decl @C5 [template = constants.%C5] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C4
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C5 {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   adapt_decl B
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C5
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
