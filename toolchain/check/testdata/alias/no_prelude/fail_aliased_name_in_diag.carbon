// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_aliased_name_in_diag.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_aliased_name_in_diag.carbon

class C {}
class D {}

alias c = C;
var d: D = {};

// CHECK:STDERR: fail_aliased_name_in_diag.carbon:[[@LINE+4]]:16: error: `Core.ImplicitAs` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: let c_var: c = d;
// CHECK:STDERR:                ^
// CHECK:STDERR:
let c_var: c = d;

// CHECK:STDOUT: --- fail_aliased_name_in_diag.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .c_var = %c_var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %C.decl [concrete = constants.%C]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %D = binding_pattern d
// CHECK:STDOUT:     %.loc15: %D = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %D = var d
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D.decl [concrete = constants.%D]
// CHECK:STDOUT:   %d: ref %D = bind_name d, %d.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c_var.patt: %C = binding_pattern c_var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [concrete = constants.%C]
// CHECK:STDOUT:   %.loc21: %C = converted @__global_init.%d.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %c_var: %C = bind_name c_var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15_13.2: init %D = class_init (), file.%d.var [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc15_1: init %D = converted %.loc15_13.1, %.loc15_13.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   assign file.%d.var, %.loc15_1
// CHECK:STDOUT:   %d.ref: ref %D = name_ref d, file.%d
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
