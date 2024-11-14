// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/fail_incomplete_element.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/fail_incomplete_element.carbon

class Incomplete;

// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE+6]]:8: error: variable has incomplete type `[Incomplete; 1]` [IncompleteTypeInVarDecl]
// CHECK:STDERR: var a: [Incomplete; 1];
// CHECK:STDERR:        ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var a: [Incomplete; 1];

var p: Incomplete* = &a[0];

// CHECK:STDOUT: --- fail_incomplete_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, %Incomplete [template]
// CHECK:STDOUT:   %.3: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .p = %p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {} {}
// CHECK:STDOUT:   %Incomplete.ref.loc19: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc19_21: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc19_22: type = array_type %.loc19_21, %Incomplete [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %Incomplete.ref.loc21: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc21: type = ptr_type %Incomplete [template = constants.%.3]
// CHECK:STDOUT:   %p.var: ref %.3 = var p
// CHECK:STDOUT:   %p: ref %.3 = bind_name p, %p.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: ref <error> = name_ref a, file.%a
// CHECK:STDOUT:   %.loc21_25: Core.IntLiteral = int_value 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc21_22: <error> = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign file.%p.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
