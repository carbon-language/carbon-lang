// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Incomplete;

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+6]]:8: ERROR: Variable has incomplete type `{.a: Incomplete}`.
// CHECK:STDERR: var s: {.a: Incomplete};
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE-5]]:1: Class was forward declared here.
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
var s: {.a: Incomplete};

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `<error>*` to `Incomplete*`.
// CHECK:STDERR: var p: Incomplete* = &s.a;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
var p: Incomplete* = &s.a;

// CHECK:STDOUT: --- fail_nested_incomplete.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: Incomplete} [template]
// CHECK:STDOUT:   %.2: type = ptr_type Incomplete [template]
// CHECK:STDOUT:   %.3: type = ptr_type <error> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.Incomplete = %Incomplete.decl, .s = %s, .p = %p}
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %Incomplete.ref.loc15: type = name_ref Incomplete, %Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc15: type = struct_type {.a: Incomplete} [template = constants.%.1]
// CHECK:STDOUT:   %s.var: ref <error> = var s
// CHECK:STDOUT:   %s: ref <error> = bind_name s, %s.var
// CHECK:STDOUT:   %Incomplete.ref.loc20: type = name_ref Incomplete, %Incomplete [template = constants.%Incomplete]
// CHECK:STDOUT:   %.loc20_18: type = ptr_type Incomplete [template = constants.%.2]
// CHECK:STDOUT:   %p.var: ref Incomplete* = var p
// CHECK:STDOUT:   %p: ref Incomplete* = bind_name p, %p.var
// CHECK:STDOUT:   %s.ref: ref <error> = name_ref s, %s
// CHECK:STDOUT:   %.loc20_22: <error>* = addr_of <error>
// CHECK:STDOUT:   assign %p.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
