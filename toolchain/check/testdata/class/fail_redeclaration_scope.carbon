// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_redeclaration_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_redeclaration_scope.carbon

class A;

class X {
  // OK, a different A.
  class A { class B; }
  class A.B {}
}

class A { class B; }

class Y {
  // CHECK:STDERR: fail_redeclaration_scope.carbon:[[@LINE+3]]:9: error: name `A` not found [NameNotFound]
  // CHECK:STDERR:   class A.B {}
  // CHECK:STDERR:         ^
  class A.B {}
}

// CHECK:STDOUT: --- fail_redeclaration_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.1: type = class_type @A.1 [template]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %A.2: type = class_type @A.2 [template]
// CHECK:STDOUT:   %B.1: type = class_type @B.1 [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %B.2: type = class_type @B.2 [template]
// CHECK:STDOUT:   %Y: type = class_type @Y [template]
// CHECK:STDOUT:   %.3: type = class_type @.1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl.loc11
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .Y = %Y.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc11: type = class_decl @A.1 [template = constants.%A.1] {} {}
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   %A.decl.loc19: type = class_decl @A.1 [template = constants.%A.1] {} {}
// CHECK:STDOUT:   %Y.decl: type = class_decl @Y [template = constants.%Y] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.1 {
// CHECK:STDOUT:   %B.decl: type = class_decl @B.2 [template = constants.%B.2] {} {}
// CHECK:STDOUT:   %.loc19: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.1
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %A.decl: type = class_decl @A.2 [template = constants.%A.2] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B.1 [template = constants.%B.1] {} {}
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .A = %A.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.2 {
// CHECK:STDOUT:   %B.decl: type = class_decl @B.1 [template = constants.%B.1] {} {}
// CHECK:STDOUT:   %.loc15: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.2
// CHECK:STDOUT:   .B = %B.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B.1 {
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B.2;
// CHECK:STDOUT:
// CHECK:STDOUT: class @Y {
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.3] {} {}
// CHECK:STDOUT:   %.loc26: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Y
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %.loc25: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
