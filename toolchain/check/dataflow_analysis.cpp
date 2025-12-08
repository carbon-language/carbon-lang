// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/dataflow_analysis.h"

#include "common/set.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Represents a single fact with two IDs.
// The meaning of id1 and id2 depends on the FactType.
// Using int32_t to store the raw index values of the various Id types.
struct Fact {
  int32_t id1;
  int32_t id2;

  friend auto operator==(const Fact& lhs, const Fact& rhs) -> bool {
    return lhs.id1 == rhs.id1 && lhs.id2 == rhs.id2;
  }
};

// Hasher for Fact to use with Carbon::Set.
inline auto CarbonHashValue(const Fact& fact, uint64_t seed) -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(fact.id1);
  hasher.HashRaw(fact.id2);
  return static_cast<HashCode>(hasher);
}

struct DataflowFacts {
  // Leader(block_id, inst_id)
  Set<Fact> leaders;
  // Edge(inst_id_from, inst_id_to)
  Set<Fact> edges;
  // BranchEdge(inst_id_from, block_id_to)
  Set<Fact> branch_edges;
  // Def(inst_id, var_id) - Definitions (VarStorage)
  Set<Fact> defs;
  // Assign(inst_id, var_id) - Assignments
  Set<Fact> assigns;
  // Use(inst_id, var_id) - Uses
  Set<Fact> uses;
  // Live(inst_id, var_id) - Variable is live at instruction
  Set<Fact> live;
};

// Recursive helper to find EntityNameIds from a pattern.
static auto CollectEntityNamesFromPattern(
    const SemIR::File& sem_ir, SemIR::InstId pattern_id,
    llvm::SmallVectorImpl<std::pair<SemIR::EntityNameId, SemIR::InstId>>& names)
    -> void {
  auto inst = sem_ir.insts().Get(pattern_id);
  if (auto var_pattern = inst.TryAs<SemIR::VarPattern>()) {
    CollectEntityNamesFromPattern(sem_ir, var_pattern->subpattern_id, names);
  } else if (auto var_param = inst.TryAs<SemIR::VarParamPattern>()) {
    CollectEntityNamesFromPattern(sem_ir, var_param->subpattern_id, names);
  } else if (auto ref_param = inst.TryAs<SemIR::RefParamPattern>()) {
    CollectEntityNamesFromPattern(sem_ir, ref_param->subpattern_id, names);
  } else if (auto val_param = inst.TryAs<SemIR::ValueParamPattern>()) {
    CollectEntityNamesFromPattern(sem_ir, val_param->subpattern_id, names);
  } else if (auto ref_bind = inst.TryAs<SemIR::RefBindingPattern>()) {
    names.push_back({ref_bind->entity_name_id, pattern_id});
  } else if (auto val_bind = inst.TryAs<SemIR::ValueBindingPattern>()) {
    names.push_back({val_bind->entity_name_id, pattern_id});
  } else if (auto tuple_pattern = inst.TryAs<SemIR::TuplePattern>()) {
    auto elements = sem_ir.inst_blocks().Get(tuple_pattern->elements_id);
    for (auto element_id : elements) {
      CollectEntityNamesFromPattern(sem_ir, element_id, names);
    }
  }
}

struct VarInfo {
  SemIR::EntityNameId entity_id;
  SemIR::InstId def_inst_id;
};

// Helper to get variable info from various instructions.
static auto GetVarInfos(const SemIR::File& sem_ir, SemIR::InstId inst_id)
    -> llvm::SmallVector<VarInfo> {
  llvm::SmallVector<VarInfo> infos;
  auto inst = sem_ir.insts().Get(inst_id);

  if (auto var_storage = inst.TryAs<SemIR::VarStorage>()) {
    if (var_storage->pattern_id.has_value()) {
      llvm::SmallVector<std::pair<SemIR::EntityNameId, SemIR::InstId>> names;
      CollectEntityNamesFromPattern(sem_ir, var_storage->pattern_id, names);
      for (auto [entity_id, def_id] : names) {
        infos.push_back({entity_id, def_id});
      }
    }
  } else if (auto ref_bind = inst.TryAs<SemIR::RefBinding>()) {
    infos.push_back({ref_bind->entity_name_id, inst_id});
  } else if (auto val_bind = inst.TryAs<SemIR::ValueBinding>()) {
    infos.push_back({val_bind->entity_name_id, inst_id});
  } else if (auto name_ref = inst.TryAs<SemIR::NameRef>()) {
    // NameRef.value_id points to the binding (RefBinding/ValueBinding).
    auto binding_id = name_ref->value_id;
    auto binding_inst = sem_ir.insts().Get(binding_id);
    if (auto ref_bind = binding_inst.TryAs<SemIR::RefBinding>()) {
      infos.push_back({ref_bind->entity_name_id, binding_id});
    } else if (auto val_bind = binding_inst.TryAs<SemIR::ValueBinding>()) {
      infos.push_back({val_bind->entity_name_id, binding_id});
    }
  }
  return infos;
}

// This builds facts needed for carrying out dataflow analysis.
static auto BuildDataflowFacts(const SemIR::File& sem_ir,
                               SemIR::FunctionId function_id) -> DataflowFacts {
  DataflowFacts facts;
  const auto& function = sem_ir.functions().Get(function_id);

  if (function.body_block_ids.empty()) {
    return facts;
  }

  // Track ref params to treat assignments as uses.
  Set<int32_t> ref_params;

  // Collect definitions from parameters.
  if (function.param_patterns_id.has_value()) {
    auto param_patterns = sem_ir.inst_blocks().Get(function.param_patterns_id);
    for (auto pattern_id : param_patterns) {
      llvm::SmallVector<std::pair<SemIR::EntityNameId, SemIR::InstId>>
          entity_names;
      CollectEntityNamesFromPattern(sem_ir, pattern_id, entity_names);
      for (auto [entity_name_id, def_inst_id] : entity_names) {
        // Use the pattern_id as the instruction ID for the definition.
        facts.defs.Insert(Fact{def_inst_id.index, entity_name_id.index});

        // Identify ref parameters.
        auto inst = sem_ir.insts().Get(pattern_id);
        if (inst.Is<SemIR::RefParamPattern>()) {
          ref_params.Insert(entity_name_id.index);
        }
      }
    }
  }

  for (const auto& block_id : function.body_block_ids) {
    const auto& block = sem_ir.inst_blocks().Get(block_id);

    // Emit leader fact for non-empty blocks.
    if (!block.empty()) {
      facts.leaders.Insert(Fact{block_id.index, block.front().index});
    }

    // First pass: identify LHS of assignments to avoid counting them as uses.
    Set<SemIR::InstId> assigned_lhs;
    for (const auto& inst_id : block) {
      auto inst = sem_ir.insts().Get(inst_id);
      if (auto assign = inst.TryAs<SemIR::Assign>()) {
        assigned_lhs.Insert(assign->lhs_id);
      }
    }

    for (size_t i = 0; i < block.size(); ++i) {
      auto inst_id = block[i];
      auto inst = sem_ir.insts().Get(inst_id);

      // Intra-block edge
      if (i + 1 < block.size()) {
        auto next_inst_id = block[i + 1];
        facts.edges.Insert(Fact{inst_id.index, next_inst_id.index});
      }

      // 1. Definition (VarStorage)
      if (inst.Is<SemIR::VarStorage>() || inst.Is<SemIR::ValueBinding>()) {
        auto var_infos = GetVarInfos(sem_ir, inst_id);
        for (auto [var_id, def_inst_id] : var_infos) {
          facts.defs.Insert(Fact{def_inst_id.index, var_id.index});
        }
      }

      // 2. Assignment
      else if (auto assign = inst.TryAs<SemIR::Assign>()) {
        auto var_infos = GetVarInfos(sem_ir, assign->lhs_id);
        for (auto [var_id, _] : var_infos) {
          facts.assigns.Insert(Fact{inst_id.index, var_id.index});
        }
      }

      // 3. Use (NameRef)
      else if (inst.Is<SemIR::NameRef>()) {
        auto var_infos = GetVarInfos(sem_ir, inst_id);
        for (auto [var_id, _] : var_infos) {
          bool is_lhs = assigned_lhs.Contains(inst_id);
          // If it's a ref parameter, assignment counts as a use.
          if (!is_lhs || ref_params.Contains(var_id.index)) {
            facts.uses.Insert(Fact{inst_id.index, var_id.index});
          }
        }
      }

      // 4. Use (ValueOfInitializer)
      //    This is used when returning a var by value.
      else if (auto val_init = inst.TryAs<SemIR::ValueOfInitializer>()) {
        auto var_infos = GetVarInfos(sem_ir, val_init->init_id);
        for (auto [var_id, _] : var_infos) {
          facts.uses.Insert(Fact{inst_id.index, var_id.index});
        }
      }

      // 5. Use (AcquireValue)
      //    This is used when converting a reference to a value (e.g. return
      //    var).
      else if (auto acquire = inst.TryAs<SemIR::AcquireValue>()) {
        auto var_infos = GetVarInfos(sem_ir, acquire->value_id);
        for (auto [var_id, _] : var_infos) {
          facts.uses.Insert(Fact{inst_id.index, var_id.index});
        }
      }

      // 6. Use (ReturnExpr)
      //    This is used when returning a var directly (e.g. with return slot).
      else if (auto ret = inst.TryAs<SemIR::ReturnExpr>()) {
        auto var_infos = GetVarInfos(sem_ir, ret->expr_id);
        for (auto [var_id, _] : var_infos) {
          facts.uses.Insert(Fact{inst_id.index, var_id.index});
        }
      }

      // 7. Use (ReturnSlot)
      //    This is used when a returned var is declared.
      else if (auto return_slot = inst.TryAs<SemIR::ReturnSlot>()) {
        auto var_infos = GetVarInfos(sem_ir, return_slot->storage_id);
        for (auto [var_id, _] : var_infos) {
          facts.uses.Insert(Fact{inst_id.index, var_id.index});
        }
      }

      // 8. Edges (Terminators)
      if (auto branch = inst.TryAs<SemIR::Branch>()) {
        facts.branch_edges.Insert(Fact{inst_id.index, branch->target_id.index});
      } else if (auto branch_if = inst.TryAs<SemIR::BranchIf>()) {
        facts.branch_edges.Insert(
            Fact{inst_id.index, branch_if->target_id.index});
      } else if (auto branch_arg = inst.TryAs<SemIR::BranchWithArg>()) {
        facts.branch_edges.Insert(
            Fact{inst_id.index, branch_arg->target_id.index});
      }
    }
  }
  return facts;
}

static auto CheckUnusedBindings(Context& context, const DataflowFacts& facts)
    -> void {
  auto& sem_ir = context.sem_ir();

  // Collect usage locations. We track the first source-location use for each
  // variable.
  llvm::DenseMap<int32_t, SemIR::InstId> first_use;
  facts.uses.ForEach([&](const Fact& use) {
    auto [it, inserted] = first_use.insert({use.id2, SemIR::InstId(use.id1)});
    if (!inserted) {
      // Keep the earliest instruction ID.
      if (use.id1 < it->second.index) {
        it->second = SemIR::InstId(use.id1);
      }
    }
  });

  // Collect definitions to diagnose.
  llvm::SmallVector<Fact> unused_defs;
  llvm::SmallVector<Fact> unused_but_used_defs;

  facts.defs.ForEach([&](const Fact& def) {
    auto var_id = def.id2;
    auto entity_name_id = SemIR::EntityNameId(var_id);
    const auto& entity_name = sem_ir.entity_names().Get(entity_name_id);

    if (first_use.find(var_id) == first_use.end()) {
      if (!entity_name.is_unused) {
        unused_defs.push_back(def);
      }
    } else {
      if (entity_name.is_unused) {
        unused_but_used_defs.push_back(def);
      }
    }
  });

  // Sort by instruction ID (location).
  auto sort_facts = [](const Fact& a, const Fact& b) { return a.id1 < b.id1; };
  std::sort(unused_defs.begin(), unused_defs.end(), sort_facts);
  std::sort(unused_but_used_defs.begin(), unused_but_used_defs.end(),
            sort_facts);

  // Emit diagnostics.
  for (const auto& def : unused_but_used_defs) {
    auto var_id = def.id2;
    auto entity_name_id = SemIR::EntityNameId(var_id);
    const auto& entity_name = sem_ir.entity_names().Get(entity_name_id);
    auto name_id = entity_name.name_id;
    llvm::StringRef name = sem_ir.names().GetFormatted(name_id);
    auto inst_id = SemIR::InstId(def.id1);
    auto loc_id = sem_ir.insts().GetCanonicalLocId(inst_id);
    CARBON_DIAGNOSTIC(UnusedButUsed, Error,
                      "variable `{0}` is marked `unused` but is used",
                      std::string);
    auto diag = context.emitter().Build(LocIdForDiagnostics(loc_id),
                                        UnusedButUsed, name.str());
    auto use_inst_id = first_use.find(var_id)->second;
    auto use_loc_id = sem_ir.insts().GetCanonicalLocId(use_inst_id);
    CARBON_DIAGNOSTIC(UnusedButUsedHere, Note, "usage is here");
    diag.Note(LocIdForDiagnostics(use_loc_id), UnusedButUsedHere);
    diag.Emit();
  }

  for (const auto& def : unused_defs) {
    auto var_id = def.id2;
    auto entity_name_id = SemIR::EntityNameId(var_id);
    const auto& entity_name = sem_ir.entity_names().Get(entity_name_id);
    auto name_id = entity_name.name_id;
    llvm::StringRef name = sem_ir.names().GetFormatted(name_id);
    auto inst_id = SemIR::InstId(def.id1);
    auto loc_id = sem_ir.insts().GetCanonicalLocId(inst_id);
    CARBON_DIAGNOSTIC(UnusedBinding, Warning, "binding `{0}` is unused",
                      std::string);
    context.emitter().Emit(LocIdForDiagnostics(loc_id), UnusedBinding,
                           name.str());
  }
}

auto RunDataflowAnalysis(Context& context, SemIR::FunctionId function_id)
    -> void {
  auto facts = BuildDataflowFacts(context.sem_ir(), function_id);
  CheckUnusedBindings(context, facts);
}

}  // namespace Carbon::Check
