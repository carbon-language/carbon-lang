// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/where_history_stack.h"

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"

namespace Carbon::Check {

auto WhereHistoryStack::IsCurrentPeriodSelfAmbiguous() -> bool {
  for (auto h : llvm::reverse(where_history_stack_)) {
    switch (h) {
      case WhereHistory::BadWherePeriodSelf:
        return true;
      case WhereHistory::GoodWherePeriodSelf:
        return false;
      case WhereHistory::PeriodSelfImpls:
        break;
    }
  }
  return false;
}

auto WhereHistoryStack::PushWhere() -> void {
  bool saw_impls = false;
  for (auto h : llvm::reverse(where_history_stack_)) {
    switch (h) {
      case WhereHistory::BadWherePeriodSelf:
        where_history_stack_.push_back(WhereHistory::BadWherePeriodSelf);
        return;
      case WhereHistory::PeriodSelfImpls:
        CARBON_CHECK(!saw_impls, "two `impls` without `where` between?");
        saw_impls = true;
        break;
      case WhereHistory::GoodWherePeriodSelf:
        if (!saw_impls) {
          // Two `where` without `.Self impls` in between. The latter one will
          // introduce a different `.Self` value.
          where_history_stack_.push_back(WhereHistory::BadWherePeriodSelf);
          return;
        }
        saw_impls = false;
    }
  }
  where_history_stack_.push_back(WhereHistory::GoodWherePeriodSelf);
}

auto WhereHistoryStack::PushPeriodSelfImplsConstraint() -> void {
  where_history_stack_.push_back(WhereHistory::PeriodSelfImpls);
}

auto WhereHistoryStack::Pop() -> void { where_history_stack_.pop_back(); }

}  // namespace Carbon::Check
