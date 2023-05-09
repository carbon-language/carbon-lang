// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: RUNTIME ERROR: /usr/local/google/home/jperkins/.cache/bazel/_bazel_jperkins/85deb7d9d96f7e0e80b42618a55969d7/execroot/carbon/bazel-out/k8-opt/bin/explorer/explorer.runfiles/carbon/explorer/data/prelude.carbon:{{.*}}: "HALLO WELT"

package ExplorerTest api;

fn Main() -> i32 {
  Assert(false, "HALLO WELT");
  return 0;
}
