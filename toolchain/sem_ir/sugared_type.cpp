// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/sugared_type.h"

#include <variant>

#include "common/check.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_categories.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

namespace {

// Searches for an instruction that describes how the type of a given
// instruction was written. See `GetSugaredTypeOfInst`.
class SugaredTypeFinder {
 public:
  explicit SugaredTypeFinder(const File* sem_ir) : sem_ir_(sem_ir) {}

  // Finds the best spelling we can for the type of `inst_id`.
  auto Find(InstId inst_id) -> TypeInstId;

 private:
  // A transformation that is applied to a type that we find in order to produce
  // the type of the instruction that the search started from.
  enum class Step {
    // Replace a pointer type `T*` with its pointee type `T`.
    Pointee,
  };

  // Walks from `inst_id` towards the instructions that determined its type,
  // until we find an instruction that tracks how its type was written, adding
  // to `steps_` as we go. Returns `None` if no such instruction is found.
  auto FindSpelledType(InstId inst_id) -> TypeInstId;

  // Finds the declared return type of `call`, as written.
  auto FindCallReturnType(Call call) -> TypeInstId;

  // Applies `step` to `type_inst_id`, desugaring it if necessary. Returns
  // `None` if the step can't be applied.
  auto ApplyStep(Step step, TypeInstId type_inst_id) -> TypeInstId;

  // Returns `operand_id` if it has type `type_id`, and `None` otherwise. This
  // is used when looking through an instruction that is expected to have the
  // same type as one of its operands: if that turns out not to hold, we stop
  // the search rather than describing the wrong type.
  auto LookThrough(TypeId type_id, InstId operand_id) -> InstId;

  const File* sem_ir_;

  // The steps to apply to the type that we find, in reverse order.
  llvm::SmallVector<Step> steps_;
};

auto SugaredTypeFinder::LookThrough(TypeId type_id, InstId operand_id)
    -> InstId {
  if (!operand_id.has_value() ||
      sem_ir_->insts().Get(operand_id).type_id() != type_id) {
    return InstId::None;
  }
  return operand_id;
}

auto SugaredTypeFinder::FindCallReturnType(Call call) -> TypeInstId {
  auto callee = GetCallee(*sem_ir_, call.callee_id);
  auto* function_callee = std::get_if<CalleeFunction>(&callee);
  if (!function_callee) {
    return TypeInstId::None;
  }

  // TODO: For a call to a generic function, substitute the call's arguments
  // into the declared return type so that we can describe how it was written.
  if (function_callee->enclosing_specific_id.has_value() ||
      function_callee->resolved_specific_id.has_value()) {
    return TypeInstId::None;
  }

  // This is `None` if no return type was declared, in which case we have no
  // spelling to offer.
  return sem_ir_->functions()
      .Get(function_callee->function_id)
      .return_type_inst_id;
}

auto SugaredTypeFinder::ApplyStep(Step step, TypeInstId type_inst_id)
    -> TypeInstId {
  switch (step) {
    case Step::Pointee: {
      auto pointer_type = sem_ir_->insts().TryGetAs<PointerType>(type_inst_id);
      if (!pointer_type) {
        // The spelling we found isn't syntactically a pointer type, for example
        // because the pointer type was named by an alias. Desugar it and try
        // again: the canonical instruction for a pointer type is a
        // `PointerType`.
        type_inst_id = sem_ir_->types().GetTypeInstId(
            sem_ir_->types().GetTypeIdForTypeInstId(type_inst_id));
        pointer_type = sem_ir_->insts().TryGetAs<PointerType>(type_inst_id);
        if (!pointer_type) {
          return TypeInstId::None;
        }
      }
      return pointer_type->pointee_id;
    }
  }
}

auto SugaredTypeFinder::FindSpelledType(InstId inst_id) -> TypeInstId {
  while (inst_id.has_value()) {
    auto inst = sem_ir_->insts().Get(inst_id);
    CARBON_KIND_SWITCH(inst) {
      // The type of a call is the declared return type of the callee.
      case CARBON_KIND(Call call): {
        return FindCallReturnType(call);
      }

      // The type of a dereference is the pointee type of the pointer.
      case CARBON_KIND(Deref deref): {
        steps_.push_back(Step::Pointee);
        inst_id = deref.pointer_id;
        continue;
      }

      // The following instructions have the same type as one of their operands,
      // so look through them to that operand.
      case CARBON_KIND(NameRef name_ref): {
        inst_id = LookThrough(inst.type_id(), name_ref.value_id);
        continue;
      }
      case CARBON_KIND_ANY(AnyBinding, binding): {
        inst_id = LookThrough(inst.type_id(), binding.value_id);
        continue;
      }
      case CARBON_KIND(AcquireValue acquire_value): {
        inst_id = LookThrough(inst.type_id(), acquire_value.value_id);
        continue;
      }
      case CARBON_KIND(Converted converted): {
        inst_id = LookThrough(inst.type_id(), converted.result_id);
        continue;
      }
      case CARBON_KIND(SpliceBlock splice_block): {
        inst_id = LookThrough(inst.type_id(), splice_block.result_id);
        continue;
      }
      case CARBON_KIND(Temporary temporary): {
        inst_id = LookThrough(inst.type_id(), temporary.init_id);
        continue;
      }
      case CARBON_KIND(ValueOfInitializer value_of_initializer): {
        inst_id = LookThrough(inst.type_id(), value_of_initializer.init_id);
        continue;
      }

      default: {
        // TODO: Handle more cases here. For example, the type of a name
        // reference to a binding or field should use the type as written in
        // the declaration of that binding or field, and the type of an index
        // into an array should be the element type as written in the array
        // type.
        return TypeInstId::None;
      }
    }
  }
  return TypeInstId::None;
}

auto SugaredTypeFinder::Find(InstId inst_id) -> TypeInstId {
  auto type_id = sem_ir_->insts().Get(inst_id).type_id();
  if (!type_id.has_value()) {
    return TypeInstId::None;
  }

  auto result = FindSpelledType(inst_id);
  for (auto step : llvm::reverse(steps_)) {
    if (!result.has_value()) {
      break;
    }
    result = ApplyStep(step, result);
  }

  if (!result.has_value()) {
    // We've no better spelling for this type, so use the canonical one.
    return sem_ir_->types().GetTypeInstId(type_id);
  }

  CARBON_CHECK(sem_ir_->types().GetTypeIdForTypeInstId(result) == type_id,
               "Spelling {0} found for the type of {1} describes type {2}, but "
               "its type is {3}",
               result, inst_id, sem_ir_->types().GetTypeIdForTypeInstId(result),
               type_id);
  return result;
}

}  // namespace

auto GetSugaredTypeOfInst(const File& sem_ir, InstId inst_id) -> TypeInstId {
  return SugaredTypeFinder(&sem_ir).Find(inst_id);
}

}  // namespace Carbon::SemIR
