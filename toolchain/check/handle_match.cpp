// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {

auto HandleMatchConditionStart(Context& context,
                               Parse::MatchConditionStartId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchConditionStart");
}

auto HandleMatchCondition(Context& context, Parse::MatchConditionId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCondition");
}

auto HandleMatchIntroducer(Context& context, Parse::MatchIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchIntroducer");
}

auto HandleMatchStatementStart(Context& context,
                               Parse::MatchStatementStartId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchStatementStart");
}

auto HandleMatchCaseIntroducer(Context& context,
                               Parse::MatchCaseIntroducerId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchCaseIntroducer");
}

auto HandleMatchCaseGuardIntroducer(Context& context,
                                    Parse::MatchCaseGuardIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseGuardIntroducer");
}

auto HandleMatchCaseGuardStart(Context& context,
                               Parse::MatchCaseGuardStartId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchCaseGuardStart");
}

auto HandleMatchCaseGuard(Context& context, Parse::MatchCaseGuardId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseGuard");
}

auto HandleMatchCaseEqualGreater(Context& context,
                                 Parse::MatchCaseEqualGreaterId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseEqualGreater");
}

auto HandleMatchCaseStart(Context& context, Parse::MatchCaseStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseStart");
}

auto HandleMatchCase(Context& context, Parse::MatchCaseId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchCase");
}

auto HandleMatchDefaultIntroducer(Context& context,
                                  Parse::MatchDefaultIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "MatchDefaultIntroducer");
}

auto HandleMatchDefaultEqualGreater(Context& context,
                                    Parse::MatchDefaultEqualGreaterId node_id)
    -> bool {
  return context.TODO(node_id, "MatchDefaultEqualGreater");
}

auto HandleMatchDefaultStart(Context& context,
                             Parse::MatchDefaultStartId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchDefaultStart");
}

auto HandleMatchDefault(Context& context, Parse::MatchDefaultId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchDefault");
}

auto HandleMatchStatement(Context& context, Parse::MatchStatementId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchStatement");
}

}  // namespace Carbon::Check
