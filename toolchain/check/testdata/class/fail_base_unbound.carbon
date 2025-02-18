// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_base_unbound.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_base_unbound.carbon

base class B {}

class C {
  extend base: B;
}

// CHECK:STDERR: fail_base_unbound.carbon:[[@LINE+4]]:12: error: expression cannot be used as a value [UseOfNonExprAsValue]
// CHECK:STDERR: let b: B = C.base;
// CHECK:STDERR:            ^~~~~~
// CHECK:STDERR:
let b: B = C.base;

// CHECK:STDOUT: --- fail_base_unbound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %B = binding_pattern b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.ref: type = name_ref B, %B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %b: %B = bind_name b, <error>
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
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc14: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.0ff [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %base.ref: %C.elem = name_ref base, @C.%.loc14 [concrete = @C.%.loc14]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
