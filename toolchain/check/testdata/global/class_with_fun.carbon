// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/global/class_with_fun.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/global/class_with_fun.carbon
class A {}

fn ret_a() -> A {
  return {};
}

var a: A = {};

// CHECK:STDOUT: --- class_with_fun.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %ret_a.type: type = fn_type @ret_a [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %ret_a: %ret_a.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.1 [template]
// CHECK:STDOUT:   %struct: %A = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .ret_a = %ret_a.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %ret_a.decl: %ret_a.type = fn_decl @ret_a [template = constants.%ret_a] {
// CHECK:STDOUT:     %A.ref.loc12: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     @ret_a.%return: ref %A = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref.loc16: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %a.var: ref %A = var a
// CHECK:STDOUT:   %a: ref %A = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ret_a() -> %return: %A {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_11.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc13_11.2: init %A = class_init (), %return [template = constants.%struct]
// CHECK:STDOUT:   %.loc13_12: init %A = converted %.loc13_11.1, %.loc13_11.2 [template = constants.%struct]
// CHECK:STDOUT:   return %.loc13_12 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc16_13.2: init %A = class_init (), file.%a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc16_14: init %A = converted %.loc16_13.1, %.loc16_13.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%a.var, %.loc16_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
