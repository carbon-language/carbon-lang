// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/deduce.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

namespace {
struct DeductionWorklist {
  auto Add(SemIR::InstId param, SemIR::InstId arg, bool needs_substitution)
      -> void {
    deductions.push_back(
        {.param = param, .arg = arg, .needs_substitution = needs_substitution});
  }

  auto AddBlock(llvm::ArrayRef<SemIR::InstId> params,
                llvm::ArrayRef<SemIR::InstId> args, bool needs_substitution)
      -> void {
    if (params.size() != args.size()) {
      return;
    }
    for (auto [param, arg] : llvm::zip_equal(params, args)) {
      Add(param, arg, needs_substitution);
    }
  }

  auto AddBlock(SemIR::InstBlockId params, llvm::ArrayRef<SemIR::InstId> args,
                bool needs_substitution) -> void {
    AddBlock(context.inst_blocks().Get(params), args, needs_substitution);
  }

  auto AddBlock(SemIR::InstBlockId params, SemIR::InstBlockId args,
                bool needs_substitution) -> void {
    AddBlock(context.inst_blocks().Get(params), context.inst_blocks().Get(args),
             needs_substitution);
  }

  struct PendingDeduction {
    SemIR::InstId param;
    SemIR::InstId arg;
    bool needs_substitution;
  };
  Context& context;
  llvm::SmallVector<PendingDeduction> deductions;
};
}  // namespace

static auto NoteGenericHere(Context& context, SemIR::GenericId generic_id,
                            Context::DiagnosticBuilder& diag) -> void {
  CARBON_DIAGNOSTIC(DeductionGenericHere, Note,
                    "While deducing parameters of generic declared here.");
  diag.Note(context.generics().Get(generic_id).decl_id, DeductionGenericHere);
}

auto DeduceGenericCallArguments(Context& context, Parse::NodeId node_id,
                                SemIR::GenericId generic_id,
                                SemIR::SpecificId enclosing_specific_id,
                                SemIR::InstBlockId implicit_params_id,
                                SemIR::InstBlockId params_id,
                                SemIR::InstId self_id,
                                llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> SemIR::SpecificId {
  DeductionWorklist worklist = {.context = context};

  // TODO: Perform deduction for type of self
  static_cast<void>(implicit_params_id);
  static_cast<void>(self_id);

  // Copy any outer generic arguments from the specified instance and prepare to
  // substitute them into the function declaration.
  llvm::SmallVector<SemIR::InstId> results;
  llvm::SmallVector<Substitution> substitutions;
  if (enclosing_specific_id.is_valid()) {
    auto args = context.inst_blocks().Get(
        context.specifics().Get(enclosing_specific_id).args_id);
    results.assign(args.begin(), args.end());

    // TODO: Subst is linear in the length of the substitutions list. Change it
    // so we can pass in an array mapping indexes to substitutions instead.
    substitutions.reserve(args.size());
    for (auto [i, subst_inst_id] : llvm::enumerate(args)) {
      substitutions.push_back(
          {.bind_id = SemIR::CompileTimeBindIndex(i),
           .replacement_id = context.constant_values().Get(subst_inst_id)});
    }
  }
  auto first_deduced_index = SemIR::CompileTimeBindIndex(results.size());

  worklist.AddBlock(params_id, arg_ids, /*needs_substitution=*/true);

  results.resize(context.inst_blocks()
                     .Get(context.generics().Get(generic_id).bindings_id)
                     .size(),
                 SemIR::InstId::Invalid);

  while (!worklist.deductions.empty()) {
    auto [param_id, arg_id, needs_substitution] =
        worklist.deductions.pop_back_val();

    // If the parameter has a symbolic type, deduce against that.
    auto param_type_id = context.insts().Get(param_id).type_id();
    if (param_type_id.AsConstantId().is_symbolic()) {
      worklist.Add(
          context.types().GetInstId(param_type_id),
          context.types().GetInstId(context.insts().Get(arg_id).type_id()),
          needs_substitution);
    }

    // If the parameter is a symbolic constant, deduce against it.
    auto param_const_id = context.constant_values().Get(param_id);
    if (!param_const_id.is_valid() || !param_const_id.is_symbolic()) {
      continue;
    }

    // If we've not yet substituted into the parameter, do so now.
    if (needs_substitution) {
      param_const_id = SubstConstant(context, param_const_id, substitutions);
      if (!param_const_id.is_valid() || !param_const_id.is_symbolic()) {
        continue;
      }
      needs_substitution = false;
    }

    CARBON_KIND_SWITCH(context.insts().Get(context.constant_values().GetInstId(
                           param_const_id))) {
      // Deducing a symbolic binding from an argument with a constant value
      // deduces the binding as having that constant value.
      case CARBON_KIND(SemIR::BindSymbolicName bind): {
        auto& entity_name = context.entity_names().Get(bind.entity_name_id);
        auto index = entity_name.bind_index;
        if (index.is_valid() && index >= first_deduced_index) {
          CARBON_CHECK(static_cast<size_t>(index.index) < results.size())
              << "Deduced value for unexpected index " << index
              << "; expected to deduce " << results.size() << " arguments.";
          auto arg_const_inst_id =
              context.constant_values().GetConstantInstId(arg_id);
          if (arg_const_inst_id.is_valid()) {
            if (results[index.index].is_valid() &&
                results[index.index] != arg_const_inst_id) {
              // TODO: Include the two different deduced values.
              CARBON_DIAGNOSTIC(DeductionInconsistent, Error,
                                "Inconsistent deductions for value of generic "
                                "parameter `{0}`.",
                                SemIR::NameId);
              auto diag = context.emitter().Build(
                  node_id, DeductionInconsistent, entity_name.name_id);
              NoteGenericHere(context, generic_id, diag);
              diag.Emit();
              return SemIR::SpecificId::Invalid;
            }
            results[index.index] = arg_const_inst_id;
          }
        }
        break;
      }

        // TODO: Handle more cases.

      default:
        break;
    }
  }

  // Check we deduced an argument value for every parameter.
  for (auto [i, deduced_arg_id] : llvm::enumerate(results)) {
    if (!deduced_arg_id.is_valid()) {
      auto binding_id = context.inst_blocks().Get(
          context.generics().Get(generic_id).bindings_id)[i];
      auto entity_name_id =
          context.insts().GetAs<SemIR::AnyBindName>(binding_id).entity_name_id;
      CARBON_DIAGNOSTIC(DeductionIncomplete, Error,
                        "Cannot deduce value for generic parameter `{0}`.",
                        SemIR::NameId);
      auto diag = context.emitter().Build(
          node_id, DeductionIncomplete,
          context.entity_names().Get(entity_name_id).name_id);
      NoteGenericHere(context, generic_id, diag);
      diag.Emit();
      return SemIR::SpecificId::Invalid;
    }
  }

  // TODO: Convert the deduced values to the types of the bindings.

  return MakeSpecific(context, generic_id,
                      context.inst_blocks().AddCanonical(results));
}

}  // namespace Carbon::Check
