// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_main.carbon

// CHECK:STDERR: fail_main.carbon:[[@LINE+4]]:1: ERROR: Imports from the current package must omit the package name.
// CHECK:STDERR: import Main;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Main;

// CHECK:STDERR: fail_main.carbon:[[@LINE+4]]:1: ERROR: Imports from the current package must omit the package name.
// CHECK:STDERR: import Main library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Main library "lib";

// --- fail_not_main.carbon

package NotMain api;

// CHECK:STDERR: fail_not_main.carbon:[[@LINE+4]]:1: ERROR: Cannot import `Main` from other packages.
// CHECK:STDERR: import Main;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Main;

// CHECK:STDERR: fail_not_main.carbon:[[@LINE+4]]:1: ERROR: Cannot import `Main` from other packages.
// CHECK:STDERR: import Main library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Main library "lib";

// --- fail_this.carbon

package This api;

// CHECK:STDERR: fail_this.carbon:[[@LINE+4]]:1: ERROR: File cannot import itself.
// CHECK:STDERR: import This;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import This;

// --- fail_this_lib.carbon

package This library "lib" api;

// CHECK:STDERR: fail_this_lib.carbon:[[@LINE+4]]:1: ERROR: File cannot import itself.
// CHECK:STDERR: import library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import library "lib";

// --- implicit_api.carbon

package Implicit api;

// --- fail_implicit.impl.carbon

package Implicit impl;

// CHECK:STDERR: fail_implicit.impl.carbon:[[@LINE+4]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import Implicit;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Implicit;

// --- implicit_lib_api.carbon

package Implicit library "lib" api;

// --- fail_implicit_lib.impl.carbon

package Implicit library "lib" impl;

// CHECK:STDERR: fail_implicit_lib.impl.carbon:[[@LINE+4]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import Implicit library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Implicit library "lib";

// --- fail_not_found.carbon
package NotFound api;

// CHECK:STDERR: fail_not_found.carbon:[[@LINE+3]]:1: ERROR: Imported API not found.
// CHECK:STDERR: import ImportNotFound;
// CHECK:STDERR: ^~~~~~
import ImportNotFound;

// CHECK:STDOUT: --- fail_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_this.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_this_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_lib_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_implicit_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_not_found.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .ImportNotFound = %ImportNotFound
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImportNotFound: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
