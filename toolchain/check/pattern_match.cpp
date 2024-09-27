// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/pattern_match.h"

#include <functional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"

namespace Carbon::Check {
namespace {

class MatchContext {
 public:
  struct StackEntry {
    SemIR::InstId pattern_id;

    // FIXME gross hack to deal with weird structure of AddrParam
    SemIR::InstId addr_pattern_id;
  };

  // FIXME make private

  llvm::SmallVector<StackEntry> stack_;

  // FIXME gross hack
  SemIR::InstId result_ = SemIR::InstId::Invalid;

  // TODO: SemIR::InstBlockId on_failure_;
};

// FIXME make this a method, and loc_id and access_kind be data members?
// FIXME Maybe eliminate loc_id and use the pattern and scrutinee locs, to avoid
// the invalid-loc hack for the function case?
auto ProcessPattern(Context& context, MatchContext& match,
                    MatchContext::StackEntry entry) {
  auto pattern = context.insts().GetWithLocId(entry.pattern_id);
  CARBON_KIND_SWITCH(pattern.inst) {
    case SemIR::BindingPattern::Kind:
    case SemIR::SymbolicBindingPattern::Kind: {
      auto binding_pattern = pattern.inst.As<SemIR::AnyBindingPattern>();
      auto bind_name = context.insts().GetAs<SemIR::AnyBindName>(
          binding_pattern.bind_name_id);
      context.inst_block_stack().AddInstId(bind_name.value_id);
      context.inst_block_stack().AddInstId(binding_pattern.bind_name_id);
      match.result_ = binding_pattern.bind_name_id;
      if (entry.addr_pattern_id.is_valid()) {
        auto addr_pattern =
            context.insts().GetAs<SemIR::AddrPattern>(entry.addr_pattern_id);
        match.result_ = context.AddInst<SemIR::AddrParam>(
            context.insts().GetLocId(entry.addr_pattern_id),
            {.type_id = addr_pattern.type_id, .inner_id = match.result_});
      }
      break;
    }
    case CARBON_KIND(SemIR::AddrPattern addr_pattern): {
      match.stack_.push_back({.pattern_id = addr_pattern.inner_id,
                              .addr_pattern_id = entry.pattern_id});
      break;
    }
    case CARBON_KIND(SemIR::ParamPattern param_pattern): {
      entry.pattern_id = param_pattern.subpattern_id;
      match.stack_.push_back(entry);
      break;
    }
    default: {
      CARBON_FATAL("Inst kind not handled: {0}", pattern.inst.kind());
    }
  }
}

auto ProcessParameters(Context& context,
                       llvm::ArrayRef<SemIR::InstId> pattern_ids)
    -> SemIR::InstBlockId {
  MatchContext match;
  std::vector<SemIR::InstId> inner_param_insts;
  inner_param_insts.reserve(pattern_ids.size());
  for (SemIR::InstId inst_id : pattern_ids) {
    match.stack_.push_back(
        {.pattern_id = inst_id, .addr_pattern_id = SemIR::InstId::Invalid});
    while (!match.stack_.empty()) {
      auto entry = match.stack_.pop_back_val();
      ProcessPattern(context, match, entry);
    }
    inner_param_insts.push_back(match.result_);
  }

  return context.inst_blocks().Add(inner_param_insts);
}

}  // namespace

auto ProcessSignature(Context& context,
                      SemIR::InstBlockId implicit_param_patterns_id,
                      SemIR::InstBlockId param_patterns_id) -> ParameterBlocks {
  auto params_id = SemIR::InstBlockId::Invalid;
  auto implicit_params_id = SemIR::InstBlockId::Invalid;

  if (implicit_param_patterns_id.is_valid()) {
    implicit_params_id = ProcessParameters(
        context, context.inst_blocks().Get(implicit_param_patterns_id));
  }

  if (param_patterns_id.is_valid()) {
    params_id = ProcessParameters(context,
                                  context.inst_blocks().Get(param_patterns_id));
  }

  return {.implicit_params_id = implicit_params_id, .params_id = params_id};
}

}  // namespace Carbon::Check
