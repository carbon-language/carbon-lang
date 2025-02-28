// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/5

library "day5_common";

import library "io_utils";

fn PageMask(page: i32) -> Core.UInt(100) {
  // TODO: return (1 as Core.UInt(100)) << page;
  return (1 as Core.UInt(100)) << (page as Core.UInt(100));
}

class Rules {
  fn Read() -> Rules {
    returned var rules: Rules;
    var i: i32 = 0;
    while (i < 100) {
      rules.disallowed_before[i] = 0;
      ++i;
    }

    var a: i32;
    var b: i32;
    while (ReadInt(&a) and ConsumeChar(0x7C) and ReadInt(&b)) {
      // TODO: rules.disallowed_before[a] |= PageMask(b);
      rules.disallowed_before[a] = rules.disallowed_before[a] | PageMask(b);
      SkipNewline();
    }
    return var;
  }

  fn IsValidOrder[self: Self](a: i32, b: i32) -> bool {
    return self.disallowed_before[b] & PageMask(a) == 0;
  }

  var disallowed_before: array(Core.UInt(100), 100);
};

class PageList {
  fn Empty() -> PageList {
    returned var me: PageList;
    me.num_pages = 0;
    return var;
  }

  fn Read() -> PageList {
    returned var me: PageList = Empty();

    var page: i32;
    if (not ReadInt(&page)) {
      return var;
    }
    me.Add(page);
    while (ConsumeChar(0x2C)) {
      ReadInt(&page);
      me.Add(page);
    }
    SkipNewline();
    return var;
  }

  fn Add[addr self: Self*](page: i32) {
    self->pages[self->num_pages] = page;
    ++self->num_pages;
  }

  fn FollowsRules[self: Self](rules: Rules) -> bool {
    var seen: Core.UInt(100) = 0;
    var i: i32 = 0;
    while (i < self.num_pages) {
      let page: i32 = self.pages[i];
      if (seen & rules.disallowed_before[page] != 0) {
        return false;
      }
      // TODO: seen |= PageMask(page);
      seen = seen | PageMask(page);
      ++i;
    }
    return true;
  }

  fn IsPossibleFirstPage[self: Self](rules: Rules, page: i32) -> bool {
    var i: i32 = 0;
    while (i < self.num_pages) {
      if (not rules.IsValidOrder(page, self.pages[i])) {
        return false;
      }
      ++i;
    }
    return true;
  }

  fn ExtractPossibleFirstPage[addr self: Self*](rules: Rules) -> i32 {
    var i: i32 = 0;
    while (i < self->num_pages) {
      var page: i32 = self->pages[i];
      if (self->IsPossibleFirstPage(rules, page)) {
        self->pages[i] = self->pages[self->num_pages - 1];
        --self->num_pages;
        return page;
      }
      ++i;
    }
    // TODO: Assert.
    return 0;
  }

  fn MiddlePage[self: Self]() -> i32 {
    return self.pages[self.num_pages / 2];
  }

  var pages: array(i32, 24);
  var num_pages: i32;
};
