// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_extend_cycle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_extend_cycle.carbon

base class A {
}

base class B {
  // This ensures that the compiler treats A as complete.
  extend base: A;
}

// CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE+7]]:1: error: redefinition of `class A` [RedeclRedef]
// CHECK:STDERR: base class A {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE-11]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: base class A {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
base class A {
  extend base: A;
  // CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE+3]]:10: error: name `C` not found [NameNotFound]
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  var c: C;
}

// CHECK:STDOUT: --- fail_extend_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.5: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [template]
// CHECK:STDOUT:   %.7: type = class_type @.1 [template]
// CHECK:STDOUT:   %.8: type = unbound_element_type %.7, %A [template]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %.decl: type = class_decl @.1 [template = constants.%.7] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc12: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %.4 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %.loc17: <witness> = complete_type_witness %.5 [template = constants.%.6]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @.1 {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc27: %.8 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %C.ref: <error> = name_ref C, <error> [template = <error>]
// CHECK:STDOUT:   %.loc31: <error> = field_decl c, element1 [template]
// CHECK:STDOUT:   %.loc32: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%.7
// CHECK:STDOUT:   .base = %.loc27
// CHECK:STDOUT:   .c = %.loc31
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
