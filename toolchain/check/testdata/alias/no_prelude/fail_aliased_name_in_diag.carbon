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

// CHECK:STDERR: fail_aliased_name_in_diag.carbon:[[@LINE+3]]:1: error: package `Core` implicitly referenced here, but not found [CoreNotFound]
// CHECK:STDERR: let c_var: c = d;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
let c_var: c = d;

// CHECK:STDOUT: --- fail_aliased_name_in_diag.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %struct: %D = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:     .c_var = @__global_init.%c_var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %c: type = bind_alias c, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %D.ref: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:   %d.var: ref %D = var d
// CHECK:STDOUT:   %d: ref %D = bind_name d, %d.var
// CHECK:STDOUT:   %c.ref: type = name_ref c, %c [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc12: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc15_13.2: init %D = class_init (), file.%d.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc15_14: init %D = converted %.loc15_13.1, %.loc15_13.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%d.var, %.loc15_14
// CHECK:STDOUT:   %d.ref: ref %D = name_ref d, file.%d
// CHECK:STDOUT:   %.loc20: %C = converted %d.ref, <error> [template = <error>]
// CHECK:STDOUT:   %c_var: %C = bind_name c_var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
