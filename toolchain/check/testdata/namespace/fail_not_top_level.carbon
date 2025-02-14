// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_not_top_level.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_not_top_level.carbon

fn F() {
  // CHECK:STDERR: fail_not_top_level.carbon:[[@LINE+4]]:3: error: `namespace` declaration not at top level [NamespaceDeclNotAtTopLevel]
  // CHECK:STDERR:   namespace N;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  namespace N;
  fn N.F() {}
}

class C {
  // CHECK:STDERR: fail_not_top_level.carbon:[[@LINE+4]]:3: error: `namespace` declaration not at top level [NamespaceDeclNotAtTopLevel]
  // CHECK:STDERR:   namespace N;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  namespace N;
  fn N.F() {}
}

interface I {
  // CHECK:STDERR: fail_not_top_level.carbon:[[@LINE+4]]:3: error: `namespace` declaration not at top level [NamespaceDeclNotAtTopLevel]
  // CHECK:STDERR:   namespace N;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  namespace N;
  fn N.I() {}
}

// CHECK:STDOUT: --- fail_not_top_level.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.bee: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.39e: %F.type.bee = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type.1cc: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.f49: %F.type.1cc = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type.733: type = facet_type <@I.2> [concrete]
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.type.d37: type = fn_type @I.1 [concrete]
// CHECK:STDOUT:   %I: %I.type.d37 = struct_value () [concrete]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.1 [concrete = constants.%F.c41] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %I.decl: type = interface_decl @I.2 [concrete = constants.%I.type.733] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I.2 {
// CHECK:STDOUT:   %Self: %I.type.733 = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: %I.type.d37 = fn_decl @I.1 [concrete = constants.%I] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .N = %N
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.1cc = fn_decl @F.3 [concrete = constants.%F.f49] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .N = %N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %N: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.bee = fn_decl @F.2 [concrete = constants.%F.39e] {} {}
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.1(@I.2.%Self: %I.type.733) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.1(constants.%Self) {}
// CHECK:STDOUT:
