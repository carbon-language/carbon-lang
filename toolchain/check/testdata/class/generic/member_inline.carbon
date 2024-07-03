// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_inline.carbon

class Class(T:! type) {
  fn F(n: T) -> T {
    return n;
  }
}

// CHECK:STDOUT: --- member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, (%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.ref.loc12_11: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc12_11 (constants.%T)]
// CHECK:STDOUT:     %n.loc12_8.1: @Class.%T.ref.loc12_11 (%T) = param n
// CHECK:STDOUT:     %n.loc12_8.2: @Class.%T.ref.loc12_11 (%T) = bind_name n, %n.loc12_8.1
// CHECK:STDOUT:     %T.ref.loc12_17: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc12_11 (constants.%T)]
// CHECK:STDOUT:     %return.var: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@Class.%n.loc12_8.2: @Class.%T.ref.loc12_11 (%T)) -> %T
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %T = name_ref n, @Class.%n.loc12_8.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
