// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- main.carbon

// CHECK:STDERR: main.carbon:[[@LINE+3]]:1: ERROR: Imports from the current package must omit the package name.
// CHECK:STDERR: import Main;
// CHECK:STDERR: ^~~~~~
import Main;

// CHECK:STDERR: main.carbon:[[@LINE+3]]:1: ERROR: Imports from the current package must omit the package name.
// CHECK:STDERR: import Main library "lib";
// CHECK:STDERR: ^~~~~~
import Main library "lib";

// --- not_main.carbon

package NotMain api;

// CHECK:STDERR: not_main.carbon:[[@LINE+3]]:1: ERROR: Cannot import `Main` from other packages.
// CHECK:STDERR: import Main;
// CHECK:STDERR: ^~~~~~
import Main;

// CHECK:STDERR: not_main.carbon:[[@LINE+3]]:1: ERROR: Cannot import `Main` from other packages.
// CHECK:STDERR: import Main library "lib";
// CHECK:STDERR: ^~~~~~
import Main library "lib";

// --- this.carbon

package This api;

// CHECK:STDERR: this.carbon:[[@LINE+3]]:1: ERROR: File cannot import itself.
// CHECK:STDERR: import This;
// CHECK:STDERR: ^~~~~~
import This;

// --- this_lib.carbon

package This library "lib" api;

// CHECK:STDERR: this_lib.carbon:[[@LINE+3]]:1: ERROR: File cannot import itself.
// CHECK:STDERR: import library "lib";
// CHECK:STDERR: ^~~~~~
import library "lib";

// --- implicit_api.carbon

package Implicit api;

// --- implicit.impl.carbon

package Implicit impl;

// CHECK:STDERR: implicit.impl.carbon:[[@LINE+3]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import Implicit;
// CHECK:STDERR: ^~~~~~
import Implicit;

// --- implicit_lib_api.carbon

package Implicit library "lib" api;

// --- implicit_lib.impl.carbon

package Implicit library "lib" impl;

// CHECK:STDERR: implicit_lib.impl.carbon:[[@LINE+3]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import Implicit library "lib";
// CHECK:STDERR: ^~~~~~
import Implicit library "lib";

// --- not_found.carbon
package NotFound api;

// CHECK:STDERR: not_found.carbon:[[@LINE+3]]:1: ERROR: Imported API not found.
// CHECK:STDERR: import ImportNotFound;
// CHECK:STDERR: ^~~~~~
import ImportNotFound;

// CHECK:STDOUT: --- main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- not_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- this.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- this_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_lib_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- not_found.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %import: <namespace> = import ir1, ir1
// CHECK:STDOUT:   %ImportNotFound: <namespace> = bind_name ImportNotFound, %import
// CHECK:STDOUT: }
// CHECK:STDOUT:
