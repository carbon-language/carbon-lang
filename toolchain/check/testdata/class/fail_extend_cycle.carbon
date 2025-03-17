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
  // CHECK:STDERR: fail_extend_cycle.carbon:[[@LINE+4]]:10: error: name `C` not found [NameNotFound]
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  // CHECK:STDERR:
  var c: C;
}

// CHECK:STDOUT: --- fail_extend_cycle.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A.466950.1: type = class_type @A.1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A.466950.1 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A.466950.1} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT:   %A.466950.2: type = class_type @A.2 [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A.466950.2, %A.466950.1 [concrete]
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
// CHECK:STDOUT:     .A = %A.decl.loc11
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl.loc11: type = class_decl @A.1 [concrete = constants.%A.466950.1] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.decl.loc26: type = class_decl @A.2 [concrete = constants.%A.466950.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.466950.1
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl.loc11 [concrete = constants.%A.466950.1]
// CHECK:STDOUT:   %.loc16: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A.2 {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl.loc11 [concrete = constants.%A.466950.1]
// CHECK:STDOUT:   %.loc27: %A.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc32_8: <error> = field_decl c, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc32_3: <error> = var_pattern %.loc32_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref <error> = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A.466950.2
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc27
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT:   .c = %.loc32_8
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
