// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_WHERE_HISTORY_STACK_H_
#define CARBON_TOOLCHAIN_CHECK_WHERE_HISTORY_STACK_H_

#include "llvm/ADT/SmallVector.h"

namespace Carbon::Check {

// Tracks the scopes where `.Self` is introduced by `where` in order to
// determine if we have two different values of `.Self` within the same type,
// which makes the value of `.Self` ambiguous in canonical values.
class WhereHistoryStack {
 public:
  auto IsCurrentPeriodSelfAmbiguous() -> bool;

  auto PushWhere() -> void;
  auto PushPeriodSelfImplsConstraint() -> void;
  auto Pop() -> void;

 private:
  enum class WhereHistory {
    GoodWherePeriodSelf,
    PeriodSelfImpls,
    BadWherePeriodSelf,
  };
  llvm::SmallVector<WhereHistory> where_history_stack_;
};

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_WHERE_HISTORY_STACK_H_
