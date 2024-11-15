// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/no_prelude/fail_export_name_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/no_prelude/fail_export_name_member.carbon

// --- a.carbon

package Foo library "[[@TEST_NAME]]";

class C {
  var n: {};
}

// --- fail_b.carbon

package Foo library "[[@TEST_NAME]]";

import library "a";

// TODO: This diagnostic doesn't clearly explain the problem. We should instead
// say something like: Only namespace-scope names can be exported.
// CHECK:STDERR: fail_b.carbon:[[@LINE+7]]:8: error: name qualifiers are only allowed for entities that provide a scope [QualifiedNameInNonScope]
// CHECK:STDERR: export C.n;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_b.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: a.carbon:4:1: note: referenced non-scope entity declared here [QualifiedNameNonScopeEntity]
// CHECK:STDERR: class C {
// CHECK:STDERR: ^~~~~~~~~
export C.n;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C, %.1 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.n: %.1} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5_11.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc5_11.2: type = converted %.loc5_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc5_8: %.2 = field_decl n, element0 [template]
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc5_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.1: type = import_ref Foo//a, inst+1, loaded [template = constants.%C]
// CHECK:STDOUT:   %import_ref.2 = import_ref Foo//a, inst+2, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Foo//a, inst+8, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = imports.%import_ref.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .n = imports.%import_ref.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
