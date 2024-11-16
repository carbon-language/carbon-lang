// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/fail_name_conflict.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/fail_name_conflict.carbon

class C {}

alias a = C;
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+7]]:5: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: var a: C = {};
// CHECK:STDERR:     ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:7: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: alias a = C;
// CHECK:STDERR:       ^
// CHECK:STDERR:
var a: C = {};

var b: C = {};
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE+6]]:7: error: duplicate name being declared in the same scope [NameDeclDuplicate]
// CHECK:STDERR: alias b = C;
// CHECK:STDERR:       ^
// CHECK:STDERR: fail_name_conflict.carbon:[[@LINE-4]]:5: note: name is previously declared here [NameDeclPrevious]
// CHECK:STDERR: var b: C = {};
// CHECK:STDERR:     ^
alias b = C;

// CHECK:STDOUT: --- fail_name_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %struct: %C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .a = %a.loc13
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %C.ref.loc13: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.loc13: type = bind_alias a, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %a.var: ref %C = var a
// CHECK:STDOUT:   %a.loc21: ref %C = bind_name a, %a.var
// CHECK:STDOUT:   %C.ref.loc23: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %b.var: ref %C = var b
// CHECK:STDOUT:   %b: ref %C = bind_name b, %b.var
// CHECK:STDOUT:   %C.ref.loc30: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:   %.loc30: type = bind_alias <invalid>, %C.decl [template = constants.%C]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc21_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc21_13.2: init %C = class_init (), file.%a.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc21_14: init %C = converted %.loc21_13.1, %.loc21_13.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%a.var, %.loc21_14
// CHECK:STDOUT:   %.loc23_13.1: %.1 = struct_literal ()
// CHECK:STDOUT:   %.loc23_13.2: init %C = class_init (), file.%b.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc23_14: init %C = converted %.loc23_13.1, %.loc23_13.2 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%b.var, %.loc23_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
