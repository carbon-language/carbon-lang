// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/deduce.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/subst.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

namespace {
// A list of pairs of (instruction from generic, corresponding instruction from
// call to of generic) for which we still need to perform deduction, along with
// methods to add and pop pending deductions from the list. Deductions are
// popped in order from most- to least-recently pushed, with the intent that
// they are visited in depth-first order, although the order is not expected to
// matter except when it influences which error is diagnosed.
class DeductionWorklist {
 public:
  explicit DeductionWorklist(Context& context) : context_(context) {}

  struct PendingDeduction {
    SemIR::InstId param;
    SemIR::InstId arg;
    bool needs_substitution;
  };

  // Adds a single (param, arg) deduction.
  auto Add(SemIR::InstId param, SemIR::InstId arg, bool needs_substitution)
      -> void {
    deductions_.push_back(
        {.param = param, .arg = arg, .needs_substitution = needs_substitution});
  }

  // Adds a list of (param, arg) deductions. These are added in reverse order so
  // they are popped in forward order.
  auto AddAll(llvm::ArrayRef<SemIR::InstId> params,
              llvm::ArrayRef<SemIR::InstId> args, bool needs_substitution)
      -> void {
    if (params.size() != args.size()) {
      // TODO: Decide whether to error on this or just treat the parameter list
      // as non-deduced. For now we treat it as non-deduced.
      return;
    }
    for (auto [param, arg] : llvm::reverse(llvm::zip_equal(params, args))) {
      Add(param, arg, needs_substitution);
    }
  }

  auto AddAll(SemIR::InstBlockId params, llvm::ArrayRef<SemIR::InstId> args,
              bool needs_substitution) -> void {
    AddAll(context_.inst_blocks().Get(params), args, needs_substitution);
  }

  auto AddAll(SemIR::InstBlockId params, SemIR::InstBlockId args,
              bool needs_substitution) -> void {
    AddAll(context_.inst_blocks().Get(params), context_.inst_blocks().Get(args),
           needs_substitution);
  }

  // Returns whether we have completed all deductions.
  auto Done() -> bool { return deductions_.empty(); }

  // Pops the next deduction. Requires `!Done()`.
  auto PopNext() -> PendingDeduction { return deductions_.pop_back_val(); }

 private:
  Context& context_;
  llvm::SmallVector<PendingDeduction> deductions_;
};

// State that is tracked throughout the deduction process.
class DeductionContext {
 public:
  DeductionContext(Context& context, SemIR::LocId loc_id,
                   SemIR::GenericId generic_id, bool diagnose)
      : context_(&context),
        loc_id_(loc_id),
        generic_id_(generic_id),
        diagnose_(diagnose) {
    CARBON_CHECK(generic_id.is_valid(),
                 "Performing deduction for non-generic entity");

    // Initialize the deduced arguments to Invalid.
    result_arg_ids_.resize(
        context.inst_blocks()
            .Get(context.generics().Get(generic_id_).bindings_id)
            .size(),
        SemIR::InstId::Invalid);
  }

  auto context() const -> Context& { return *context_; }
  auto worklist() -> DeductionWorklist& { return worklist_; }

  // Adds the arguments from the given specific as known enclosing generic
  // arguments that will not be deduced.
  auto AddEnclosingArgs(SemIR::SpecificId enclosing_specific_id) -> void;

  // Performs all deductions in the deduction worklist. Returns whether
  // deduction succeeded.
  auto Deduce() -> bool;

  // Returns whether every generic parameter has a corresponding deduced generic
  // argument. If not, issues a suitable diagnostic.
  auto CheckDeductionIsComplete() -> bool;

  // Forms a specific corresponding to the deduced generic with the deduced
  // argument list.
  auto MakeSpecific() -> SemIR::SpecificId;

 private:
  Context* context_;
  SemIR::LocId loc_id_;
  SemIR::GenericId generic_id_;
  bool diagnose_;

  DeductionWorklist worklist_ = DeductionWorklist(*context_);

  llvm::SmallVector<SemIR::InstId> result_arg_ids_;
  llvm::SmallVector<Substitution> substitutions_;
  SemIR::CompileTimeBindIndex first_deduced_index_ =
      SemIR::CompileTimeBindIndex(0);
};

}  // namespace

static auto NoteGenericHere(Context& context, SemIR::GenericId generic_id,
                            Context::DiagnosticBuilder& diag) -> void {
  CARBON_DIAGNOSTIC(DeductionGenericHere, Note,
                    "while deducing parameters of generic declared here");
  diag.Note(context.generics().Get(generic_id).decl_id, DeductionGenericHere);
}

auto DeductionContext::AddEnclosingArgs(SemIR::SpecificId enclosing_specific_id)
    -> void {
  if (!enclosing_specific_id.is_valid()) {
    return;
  }
  // Copy any outer generic arguments from the specified instance and prepare to
  // substitute them into the function declaration.
  auto args = context().inst_blocks().Get(
      context().specifics().Get(enclosing_specific_id).args_id);
  std::copy(args.begin(), args.end(), result_arg_ids_.begin());

  // TODO: Subst is linear in the length of the substitutions list. Change
  // it so we can pass in an array mapping indexes to substitutions instead.
  substitutions_.reserve(args.size());
  for (auto [i, subst_inst_id] : llvm::enumerate(args)) {
    substitutions_.push_back(
        {.bind_id = SemIR::CompileTimeBindIndex(i),
         .replacement_id = context().constant_values().Get(subst_inst_id)});
  }
  first_deduced_index_ = SemIR::CompileTimeBindIndex(args.size());
}

auto DeductionContext::Deduce() -> bool {
  while (!worklist_.Done()) {
    auto [param_id, arg_id, needs_substitution] = worklist_.PopNext();

    // If the parameter has a symbolic type, deduce against that.
    auto param_type_id = context().insts().Get(param_id).type_id();
    if (param_type_id.AsConstantId().is_symbolic()) {
      worklist_.Add(
          context().types().GetInstId(param_type_id),
          context().types().GetInstId(context().insts().Get(arg_id).type_id()),
          needs_substitution);
    } else {
      // The argument needs to have the same type as the parameter.
      // TODO: Suppress diagnostics here if diagnose_ is false.
      DiagnosticAnnotationScope annotate_diagnostics(
          &context().emitter(), [&](auto& builder) {
            if (auto param =
                    context().insts().TryGetAs<SemIR::BindSymbolicName>(
                        param_id)) {
              CARBON_DIAGNOSTIC(
                  InitializingGenericParam, Note,
                  "initializing generic parameter `{0}` declared here",
                  SemIR::NameId);
              builder.Note(
                  param_id, InitializingGenericParam,
                  context().entity_names().Get(param->entity_name_id).name_id);
            }
          });
      arg_id = ConvertToValueOfType(context(), loc_id_, arg_id, param_type_id);
      if (arg_id == SemIR::InstId::BuiltinError) {
        return false;
      }
    }

    // If the parameter is a symbolic constant, deduce against it.
    auto param_const_id = context().constant_values().Get(param_id);
    if (!param_const_id.is_valid() || !param_const_id.is_symbolic()) {
      continue;
    }

    // If we've not yet substituted into the parameter, do so now.
    if (needs_substitution) {
      param_const_id = SubstConstant(context(), param_const_id, substitutions_);
      if (!param_const_id.is_valid() || !param_const_id.is_symbolic()) {
        continue;
      }
      needs_substitution = false;
    }

    CARBON_KIND_SWITCH(context().insts().Get(
                           context().constant_values().GetInstId(
                               param_const_id))) {
      // Deducing a symbolic binding from an argument with a constant value
      // deduces the binding as having that constant value.
      case CARBON_KIND(SemIR::BindSymbolicName bind): {
        auto& entity_name = context().entity_names().Get(bind.entity_name_id);
        auto index = entity_name.bind_index;
        if (index.is_valid() && index >= first_deduced_index_) {
          CARBON_CHECK(
              static_cast<size_t>(index.index) < result_arg_ids_.size(),
              "Deduced value for unexpected index {0}; expected to "
              "deduce {1} arguments.",
              index, result_arg_ids_.size());
          auto arg_const_inst_id =
              context().constant_values().GetConstantInstId(arg_id);
          if (arg_const_inst_id.is_valid()) {
            if (result_arg_ids_[index.index].is_valid() &&
                result_arg_ids_[index.index] != arg_const_inst_id) {
              if (diagnose_) {
                // TODO: Include the two different deduced values.
                CARBON_DIAGNOSTIC(
                    DeductionInconsistent, Error,
                    "inconsistent deductions for value of generic "
                    "parameter `{0}`",
                    SemIR::NameId);
                auto diag = context().emitter().Build(
                    loc_id_, DeductionInconsistent, entity_name.name_id);
                NoteGenericHere(context(), generic_id_, diag);
                diag.Emit();
              }
              return false;
            }
            result_arg_ids_[index.index] = arg_const_inst_id;
          }
        }
        break;
      }

        // TODO: Handle more cases.

      default:
        break;
    }
  }

  return true;
}

auto DeductionContext::CheckDeductionIsComplete() -> bool {
  // Check we deduced an argument value for every parameter.
  for (auto [i, deduced_arg_id] :
       llvm::enumerate(llvm::ArrayRef(result_arg_ids_)
                           .drop_front(first_deduced_index_.index))) {
    if (!deduced_arg_id.is_valid()) {
      if (diagnose_) {
        auto binding_index = first_deduced_index_.index + i;
        auto binding_id = context().inst_blocks().Get(
            context().generics().Get(generic_id_).bindings_id)[binding_index];
        auto entity_name_id = context()
                                  .insts()
                                  .GetAs<SemIR::AnyBindName>(binding_id)
                                  .entity_name_id;
        CARBON_DIAGNOSTIC(DeductionIncomplete, Error,
                          "cannot deduce value for generic parameter `{0}`",
                          SemIR::NameId);
        auto diag = context().emitter().Build(
            loc_id_, DeductionIncomplete,
            context().entity_names().Get(entity_name_id).name_id);
        NoteGenericHere(context(), generic_id_, diag);
        diag.Emit();
      }
      return false;
    }
  }

  return true;
}

auto DeductionContext::MakeSpecific() -> SemIR::SpecificId {
  // TODO: Convert the deduced values to the types of the bindings.

  return Check::MakeSpecific(
      context(), generic_id_,
      context().inst_blocks().AddCanonical(result_arg_ids_));
}

auto DeduceGenericCallArguments(
    Context& context, SemIR::LocId loc_id, SemIR::GenericId generic_id,
    SemIR::SpecificId enclosing_specific_id,
    [[maybe_unused]] SemIR::InstBlockId implicit_params_id,
    SemIR::InstBlockId params_id, [[maybe_unused]] SemIR::InstId self_id,
    llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::SpecificId {
  DeductionContext deduction(context, loc_id, generic_id, /*diagnose=*/true);
  deduction.AddEnclosingArgs(enclosing_specific_id);

  // Prepare to perform deduction of the explicit parameters against their
  // arguments.
  // TODO: Also perform deduction for type of self.
  deduction.worklist().AddAll(params_id, arg_ids, /*needs_substitution=*/true);

  if (!deduction.Deduce() || !deduction.CheckDeductionIsComplete()) {
    return SemIR::SpecificId::Invalid;
  }

  return deduction.MakeSpecific();
}

// Deduces the impl arguments to use in a use of a parameterized impl. Returns
// `Invalid` if deduction fails.
auto DeduceImplArguments(Context& context, SemIR::LocId loc_id,
                         const SemIR::Impl& impl, SemIR::ConstantId self_id,
                         SemIR::ConstantId constraint_id) -> SemIR::SpecificId {
  DeductionContext deduction(context, loc_id, impl.generic_id,
                             /*diagnose=*/false);

  // Prepare to perform deduction of the type and interface.
  deduction.worklist().Add(impl.self_id,
                           context.constant_values().GetInstId(self_id),
                           /*needs_substitution=*/false);
  deduction.worklist().Add(impl.constraint_id,
                           context.constant_values().GetInstId(constraint_id),
                           /*needs_substitution=*/false);

  if (!deduction.Deduce() || !deduction.CheckDeductionIsComplete()) {
    return SemIR::SpecificId::Invalid;
  }

  return deduction.MakeSpecific();
}

}  // namespace Carbon::Check
