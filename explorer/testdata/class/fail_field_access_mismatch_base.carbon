// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

base class Point {
  var x: i32;
  var y: i32;
}

class Point3D extends Point {
  var z: i32;
}

fn Main() -> i32 {
  var p: Point3D = {.base={.x = 1, .y = 2}, .z = 3};
  // CHECK:STDERR: COMPILATION ERROR: {{.*}}/explorer/testdata/class/fail_field_access_mismatch_base.carbon:[[@LINE+1]]: class Point3D does not have a field named a
  return p.a;
}
