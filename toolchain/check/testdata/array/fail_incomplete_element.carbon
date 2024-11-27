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

// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE+7]]:8: error: binding pattern has incomplete type `[Incomplete; 1]` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var a: array(Incomplete, 1);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete_element.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: array(Incomplete, 1);

var p: Incomplete* = &a[0];

// CHECK:STDOUT: --- fail_incomplete_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %Incomplete [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Incomplete [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
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
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .p = %p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [concrete = constants.%Incomplete] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %.loc20_1: <error> = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %.loc20_27: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %Incomplete.ref.loc20: type = name_ref Incomplete, %Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %Incomplete [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: <error> = bind_name a, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %.loc22_1: %ptr = var_pattern %p.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr = var p
// CHECK:STDOUT:   %.loc22_18: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Incomplete.ref.loc22: type = name_ref Incomplete, %Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:     %ptr: type = ptr_type %Incomplete [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr = bind_name p, %p.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, file.%a
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %addr: <error> = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   assign file.%p.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
