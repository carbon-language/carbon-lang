// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// RUN: %{not} %{explorer-run}
// RUN: %{not} %{explorer-run-trace}

package ExplorerTest api;

fn Main() -> i32 {
  // CHECK:STDERR: SYNTAX ERROR: {{.*}}/explorer/testdata/string/fail_raw_block_more_hash_tags_on_left.carbon:[[@LINE+1]]: Unexpected end of file
  var s: String = ##'''
    error: there are more #s on the left than the right.
  '''#;

  return 0;
}
