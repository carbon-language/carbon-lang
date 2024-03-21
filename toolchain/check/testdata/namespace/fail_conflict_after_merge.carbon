// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- namespace.carbon

package Example library "namespace" api;

namespace NS;

// --- fail_conflict.carbon

package Example api;

import library "namespace";

// The conflict diagnostics should point at this declaration rather than the
// imported declaration.
namespace NS;

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+7]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-5]]:1: Name is previously declared here.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn NS();

// The second conflict diagnostic should still point at the earlier declaration;
// we don't move it.
namespace NS;

// CHECK:STDERR: fail_conflict.carbon:[[@LINE+6]]:1: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn NS();
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_conflict.carbon:[[@LINE-18]]:1: Name is previously declared here.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
fn NS();

// CHECK:STDOUT: --- namespace.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NS: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .NS = %NS
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <namespace> = import_ref ir1, inst+1, used
// CHECK:STDOUT:   %NS: <namespace> = namespace %import_ref, [template] {}
// CHECK:STDOUT:   %.loc8: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc17: <function> = fn_decl @.1 [template] {}
// CHECK:STDOUT:   %.loc21: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc29: <function> = fn_decl @.2 [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2();
// CHECK:STDOUT:
