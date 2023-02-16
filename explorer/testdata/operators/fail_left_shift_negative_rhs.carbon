// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This won't auto-update because it uses a regex for the prelude line number,
// so that it doesn't need to be updated on every prelude change.
//
// This can't be autoupdated because the error line comes form a different file.
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}
// CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/data/prelude.carbon:{{.*}}: Integer overflow

package ExplorerTest api;

fn Main() -> i32 {
  var a: auto = 5 << -1;
  return 0;
}
