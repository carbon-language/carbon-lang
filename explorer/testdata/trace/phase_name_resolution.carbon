// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package ExplorerTest api;

interface TestInterface {}

fn Main() -> i32 {
  return 0;
}

// Place checks after code so that line numbers are stable, reducing merge
// conflicts.
// ARGS: --trace_file=- --trace_phase=name_resolution %s
// AUTOUPDATE

// CHECK:STDOUT: * * * * * * * * * *  resolving names  * * * * * * * * * *
// CHECK:STDOUT: ---------------------------------------------------------
// CHECK:STDOUT: ==> declared `TestInterface` as `interface TestInterface` in `package` (phase_name_resolution.carbon:7)
// CHECK:STDOUT: ==> declared `Main` as `fn Main` in `package` (phase_name_resolution.carbon:11)
// CHECK:STDOUT: ->> resolving decl `interface TestInterface` (phase_name_resolution.carbon:7)
// CHECK:STDOUT: ==> marked `TestInterface` declared but not usable in `package`
// CHECK:STDOUT: ==> marked `TestInterface` usable in `package`
// CHECK:STDOUT: ==> declared `Self` as `Self` in `interface TestInterface` (phase_name_resolution.carbon:7)
// CHECK:STDOUT: <<- finished resolving decl `interface TestInterface` (phase_name_resolution.carbon:7)
// CHECK:STDOUT: ->> resolving decl `fn Main` (phase_name_resolution.carbon:11)
// CHECK:STDOUT: ==> marked `Main` declared but not usable in `package`
// CHECK:STDOUT: ==> marked `Main` usable in `package`
// CHECK:STDOUT: ->> resolving stmt `{ ... }` (phase_name_resolution.carbon:11)
// CHECK:STDOUT: ->> resolving stmt `return ...;` (phase_name_resolution.carbon:10)
// CHECK:STDOUT: <<- finished resolving stmt `return ...;` (phase_name_resolution.carbon:10)
// CHECK:STDOUT: <<- finished resolving stmt `{ ... }` (phase_name_resolution.carbon:11)
// CHECK:STDOUT: <<- finished resolving decl `fn Main` (phase_name_resolution.carbon:11)
// CHECK:STDOUT: ==> resolved `Main` as `fn Main` in `package` (<Main()>:0)
// CHECK:STDOUT: result: 0
