// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/fail_nested_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/fail_nested_incomplete.carbon

class Incomplete;

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+7]]:8: error: binding pattern has incomplete type `(i32, Incomplete)` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var t: (i32, Incomplete);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var t: (i32, Incomplete);

var p: Incomplete* = &t[1];

// CHECK:STDOUT: --- fail_nested_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.8a4: type = tuple_type (%i32, %Incomplete) [template]
// CHECK:STDOUT:   %ptr: type = ptr_type %Incomplete [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .t = %t
// CHECK:STDOUT:     .p = %p
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %t.patt: <error> = binding_pattern t
// CHECK:STDOUT:     %.loc20_1: <error> = var_pattern %t.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t.var: ref <error> = var t
// CHECK:STDOUT:   %.loc20_24.1: type = splice_block %.loc20_24.3 [template = constants.%tuple.type.8a4] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %Incomplete.ref.loc20: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %.loc20_24.2: %tuple.type.24b = tuple_literal (%i32, %Incomplete.ref.loc20)
// CHECK:STDOUT:     %.loc20_24.3: type = converted %.loc20_24.2, constants.%tuple.type.8a4 [template = constants.%tuple.type.8a4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %t: <error> = bind_name t, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %ptr = binding_pattern p
// CHECK:STDOUT:     %.loc22_1: %ptr = var_pattern %p.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.var: ref %ptr = var p
// CHECK:STDOUT:   %.loc22_18: type = splice_block %ptr [template = constants.%ptr] {
// CHECK:STDOUT:     %Incomplete.ref.loc22: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %ptr: type = ptr_type %Incomplete [template = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: ref %ptr = bind_name p, %p.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %t.ref: <error> = name_ref t, file.%t
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   %addr: <error> = addr_of <error> [template = <error>]
// CHECK:STDOUT:   assign file.%p.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
