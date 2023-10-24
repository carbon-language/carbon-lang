// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Incomplete;

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+6]]:23: ERROR: Variable has incomplete type `{.a: Incomplete}`.
// CHECK:STDERR: var s: {.a: Incomplete};
// CHECK:STDERR:                       ^
// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE-5]]:1: Class was forward declared here.
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^
var s: {.a: Incomplete};

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+3]]:26: ERROR: Cannot implicitly convert from `<error>*` to `Incomplete*`.
// CHECK:STDERR: var p: Incomplete* = &s.a;
// CHECK:STDERR:                          ^
var p: Incomplete* = &s.a;

// CHECK:STDOUT: file "fail_nested_incomplete.carbon" {
// CHECK:STDOUT:   class_declaration @Incomplete, ()
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete
// CHECK:STDOUT:   %Incomplete.ref.loc15: type = name_reference "Incomplete", %Incomplete
// CHECK:STDOUT:   %.loc15: type = struct_type {.a: Incomplete}
// CHECK:STDOUT:   %s.loc15_5.1: ref <error> = var "s"
// CHECK:STDOUT:   %s.loc15_5.2: ref <error> = bind_name "s", %s.loc15_5.1
// CHECK:STDOUT:   %Incomplete.ref.loc20: type = name_reference "Incomplete", %Incomplete
// CHECK:STDOUT:   %.loc20_18: type = ptr_type Incomplete
// CHECK:STDOUT:   %p.loc20_5.1: ref Incomplete* = var "p"
// CHECK:STDOUT:   %p.loc20_5.2: ref Incomplete* = bind_name "p", %p.loc20_5.1
// CHECK:STDOUT:   %s.ref: ref <error> = name_reference "s", %s.loc15_5.2
// CHECK:STDOUT:   %.loc20_22.1: type = ptr_type <error>
// CHECK:STDOUT:   %.loc20_22.2: <error>* = address_of <error>
// CHECK:STDOUT:   assign %p.loc20_5.1, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
