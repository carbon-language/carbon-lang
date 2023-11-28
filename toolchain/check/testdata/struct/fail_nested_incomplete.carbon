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
// CHECK:STDERR: ^~~~~
var s: {.a: Incomplete};

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+3]]:1: ERROR: Cannot implicitly convert from `<error>*` to `Incomplete*`.
// CHECK:STDERR: var p: Incomplete* = &s.a;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
var p: Incomplete* = &s.a;

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc20: type = ptr_type <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_nested_incomplete.carbon" {
// CHECK:STDOUT:   class_decl @Incomplete, ()
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete
// CHECK:STDOUT:   %Incomplete.ref.loc15: type = name_ref Incomplete, %Incomplete
// CHECK:STDOUT:   %.loc15: type = struct_type {.a: Incomplete}
// CHECK:STDOUT:   %s.var: ref <error> = var s
// CHECK:STDOUT:   %s: ref <error> = bind_name s, %s.var
// CHECK:STDOUT:   %Incomplete.ref.loc20: type = name_ref Incomplete, %Incomplete
// CHECK:STDOUT:   %.loc20_18: type = ptr_type Incomplete
// CHECK:STDOUT:   %p.var: ref Incomplete* = var p
// CHECK:STDOUT:   %p: ref Incomplete* = bind_name p, %p.var
// CHECK:STDOUT:   %s.ref: ref <error> = name_ref s, %s
// CHECK:STDOUT:   %.loc20_22: <error>* = address_of <error>
// CHECK:STDOUT:   assign %p.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
