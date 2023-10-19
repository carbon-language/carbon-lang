// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Incomplete;

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+6]]:24: ERROR: Variable has incomplete type `(i32, Incomplete)`.
// CHECK:STDERR: var t: (i32, Incomplete);
// CHECK:STDERR:                        ^
// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE-5]]:1: Class was forward declared here.
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^
var t: (i32, Incomplete);

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+3]]:27: ERROR: Cannot implicitly convert from `<error>*` to `Incomplete*`.
// CHECK:STDERR: var p: Incomplete* = &t[1];
// CHECK:STDERR:                           ^
var p: Incomplete* = &t[1];

// CHECK:STDOUT: file "fail_nested_incomplete.carbon" {
// CHECK:STDOUT:   %Incomplete: type = class_declaration @Incomplete, ()
// CHECK:STDOUT:   %Incomplete.ref.loc15: type = name_reference "Incomplete", %Incomplete
// CHECK:STDOUT:   %.loc15_24.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc15_24.2: (type, type) = tuple_literal (i32, %Incomplete.ref.loc15)
// CHECK:STDOUT:   %.loc15_24.3: type = tuple_type (i32, Incomplete)
// CHECK:STDOUT:   %t: ref <error> = var "t"
// CHECK:STDOUT:   %Incomplete.ref.loc20: type = name_reference "Incomplete", %Incomplete
// CHECK:STDOUT:   %.loc20_18: type = ptr_type Incomplete
// CHECK:STDOUT:   %p: ref Incomplete* = var "p"
// CHECK:STDOUT:   %t.ref: ref <error> = name_reference "t", %t
// CHECK:STDOUT:   %.loc20_25: i32 = int_literal 1
// CHECK:STDOUT:   %.loc20_22.1: type = ptr_type <error>
// CHECK:STDOUT:   %.loc20_22.2: <error>* = address_of <error>
// CHECK:STDOUT:   assign %p, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
