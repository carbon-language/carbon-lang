// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This won't auto-update because it uses a regex for the prelude line number,
// so that it doesn't need to be updated on every prelude change.
//
// This can't be autoupdated because the error line comes form a different file.
// NOAUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}
// CHECK:STDERR: RUNTIME ERROR: {{.*}}/explorer/data/prelude.carbon:{{.*}}: "HALLO WELT"

package ExplorerTest api;

fn Main() -> i32 {
  Assert(true, "HALLO WELT");
  Assert(false, "HALLO WELT");
  return 0;
}
