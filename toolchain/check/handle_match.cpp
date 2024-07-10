// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/handle.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::MatchConditionStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchConditionStart");
}

auto HandleParseNode(Context& context, Parse::MatchConditionId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCondition");
}

auto HandleParseNode(Context& context, Parse::MatchIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchIntroducer");
}

auto HandleParseNode(Context& context, Parse::MatchStatementStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchStatementStart");
}

auto HandleParseNode(Context& context, Parse::MatchCaseIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseIntroducer");
}

auto HandleParseNode(Context& context,
                     Parse::MatchCaseGuardIntroducerId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchCaseGuardIntroducer");
}

auto HandleParseNode(Context& context, Parse::MatchCaseGuardStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseGuardStart");
}

auto HandleParseNode(Context& context, Parse::MatchCaseGuardId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseGuard");
}

auto HandleParseNode(Context& context, Parse::MatchCaseEqualGreaterId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseEqualGreater");
}

auto HandleParseNode(Context& context, Parse::MatchCaseStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchCaseStart");
}

auto HandleParseNode(Context& context, Parse::MatchCaseId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchCase");
}

auto HandleParseNode(Context& context, Parse::MatchDefaultIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "MatchDefaultIntroducer");
}

auto HandleParseNode(Context& context,
                     Parse::MatchDefaultEqualGreaterId node_id) -> bool {
  return context.TODO(node_id, "MatchDefaultEqualGreater");
}

auto HandleParseNode(Context& context, Parse::MatchDefaultStartId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchDefaultStart");
}

auto HandleParseNode(Context& context, Parse::MatchDefaultId node_id) -> bool {
  return context.TODO(node_id, "HandleMatchDefault");
}

auto HandleParseNode(Context& context, Parse::MatchStatementId node_id)
    -> bool {
  return context.TODO(node_id, "HandleMatchStatement");
}

}  // namespace Carbon::Check
