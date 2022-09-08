// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s
// CHECK: ASSERTION ERROR: {{.*}}/explorer/testdata/assert/assert.carbon:16: "HALLO WELT"

package ExplorerTest api;

fn Main() -> i32 {
  __intrinsic_assert(true, "HALLO WELT");
  __intrinsic_assert(false, "HALLO WELT");
  return 0;
}
