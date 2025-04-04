// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/no_prelude/fail_nested_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/no_prelude/fail_nested_incomplete.carbon

class Incomplete;

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+7]]:8: error: binding pattern has incomplete type `{.a: Incomplete}` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var s: {.a: Incomplete};
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var s: {.a: Incomplete};

var p: Incomplete* = &s.a;

// CHECK:STDOUT: --- fail_nested_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %Incomplete} [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %Incomplete [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .s = %s
// CHECK:STDOUT:     .p = %p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [concrete = constants.%Incomplete] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %s.patt: <error> = binding_pattern s
// CHECK:STDOUT:     %.loc20_1: <error> = var_pattern %s.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s.var: ref <error> = var s
// CHECK:STDOUT:   %.loc20_23: type = splice_block %struct_type.a [concrete = constants.%struct_type.a] {
// CHECK:STDOUT:     %Incomplete.ref.loc20: type = name_ref Incomplete, %Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %Incomplete} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %s: <error> = bind_name s, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %.loc22_1: %ptr = var_pattern %p.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr = var p
// CHECK:STDOUT:   %.loc22_18: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %Incomplete.ref.loc22: type = name_ref Incomplete, %Incomplete.decl [concrete = constants.%Incomplete]
// CHECK:STDOUT:     %ptr: type = ptr_type %Incomplete.ref.loc22 [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr = bind_name p, %p.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %s.ref: <error> = name_ref s, file.%s [concrete = <error>]
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: <error> = addr_of %a.ref [concrete = <error>]
// CHECK:STDOUT:   assign file.%p.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
