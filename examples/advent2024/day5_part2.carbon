// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/5

import Core library "io";

import library "day5_common";
import library "io_utils";

fn Run() {
  var rules: Rules = Rules.Read();
  SkipNewline();
  var total: i32 = 0;
  while (true) {
    var page_list: PageList = PageList.Read();
    if (page_list.num_pages == 0) {
      break;
    }
    if (page_list.FollowsRules(rules)) {
      continue;
    }
    var new_page_list: PageList = PageList.Empty();
    while (page_list.num_pages != 0) {
      new_page_list.Add(page_list.ExtractPossibleFirstPage(rules));
    }
    total += new_page_list.MiddlePage();
  }
  Core.Print(total);
}
