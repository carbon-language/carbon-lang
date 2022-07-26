// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: %{explorer} %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes=false %s
// RUN: %{explorer} --parser_debug --trace_file=- %s 2>&1 | \
// RUN:   %{FileCheck} --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: %{explorer} %s

package ExplorerTest api;

fn Main() -> i32 {
  var empty: {} = {};
  empty = {};
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/struct/empty.carbon:[[@LINE+2]]: {} is not equality comparable with {}:
  // CHECK: COMPILATION ERROR: {{.*}}/explorer/testdata/struct/empty.carbon:[[@LINE+1]]: could not find implementation of interface Eq for {}
  if (not (empty == {})) {
    return 1;
  }
  match (empty) {
    case {} => {
      return 0;
    }
  }
  return 1;
}
