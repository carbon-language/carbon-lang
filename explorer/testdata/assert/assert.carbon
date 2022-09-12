// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This won't auto-update because it uses a regex for the prelude line number,
// so that it doesn't need to be updated on every prelude change.
//
// RUN: %{not} %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{not} %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// NOAUTOUPDATE
// CHECK: RUNTIME ERROR: {{.*}}/explorer/data/prelude.carbon:{{.*}}: "HALLO WELT"

package ExplorerTest api;

fn Main() -> i32 {
  Assert(true, "HALLO WELT");
  Assert(false, "HALLO WELT");
  return 0;
}
