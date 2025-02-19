// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/constant.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Value.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

// Context and shared functionality for lowering constant values.
class ConstantContext {
 public:
  explicit ConstantContext(FileContext& file_context,
                           llvm::MutableArrayRef<llvm::Constant*> constants)
      : file_context_(&file_context), constants_(constants) {}

  // Gets the lowered constant value for an instruction, which must have a
  // constant value that has already been lowered. Returns nullptr if it
  // hasn't been, in which case it should not be needed.
  auto GetConstant(SemIR::InstId inst_id) const -> llvm::Constant* {
    return GetConstant(file_context_->sem_ir().constant_values().Get(inst_id));
  }

  // Gets the lowered constant value for a constant that has already been
  // lowered. Returns nullptr if it hasn't been, in which case it should not be
  // needed.
  auto GetConstant(SemIR::ConstantId const_id) const -> llvm::Constant* {
    CARBON_CHECK(const_id.is_concrete(), "Unexpected constant ID {0}",
                 const_id);
    auto inst_id =
        file_context_->sem_ir().constant_values().GetInstId(const_id);
    if (inst_id.index > last_lowered_constant_index_) {
      // This constant hasn't been lowered.
      return nullptr;
    }
    CARBON_CHECK(inst_id.index >= 0);
    return constants_[inst_id.index];
  }

  // Returns a constant for the case of a value that should never be used.
  auto GetUnusedConstant(SemIR::TypeId /*type_id*/) const -> llvm::Constant* {
    // TODO: Consider using a poison value of the appropriate type.
    return nullptr;
  }

  // Gets the value to use for an integer literal.
  auto GetIntLiteralAsValue() const -> llvm::Constant* {
    return file_context_->GetIntLiteralAsValue();
  }

  // Gets a callable's function. Returns nullptr for a builtin.
  auto GetFunction(SemIR::FunctionId function_id) -> llvm::Function* {
    return file_context_->GetFunction(function_id);
  }

  // Returns a lowered type for the given type_id.
  auto GetType(SemIR::TypeId type_id) const -> llvm::Type* {
    return file_context_->GetType(type_id);
  }

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() const -> llvm::Constant* {
    return file_context_->GetTypeAsValue();
  }

  // Sets the index of the constant we most recently lowered. This is used to
  // check we don't look at constants that we've not lowered yet.
  auto SetLastLoweredConstantIndex(int32_t index) {
    last_lowered_constant_index_ = index;
  }

  auto llvm_context() const -> llvm::LLVMContext& {
    return file_context_->llvm_context();
  }
  auto llvm_module() const -> llvm::Module& {
    return file_context_->llvm_module();
  }
  auto sem_ir() const -> const SemIR::File& { return file_context_->sem_ir(); }

 private:
  FileContext* file_context_;
  llvm::MutableArrayRef<llvm::Constant*> constants_;
  int32_t last_lowered_constant_index_ = -1;
};

// Emits an aggregate constant of LLVM type `Type` whose elements are the
// contents of `refs_id`.
template <typename ConstantType, typename Type>
static auto EmitAggregateConstant(ConstantContext& context,
                                  SemIR::InstBlockId refs_id, Type* llvm_type)
    -> llvm::Constant* {
  auto refs = context.sem_ir().inst_blocks().Get(refs_id);
  llvm::SmallVector<llvm::Constant*> elements;
  elements.reserve(refs.size());
  for (auto ref : refs) {
    elements.push_back(context.GetConstant(ref));
  }

  return ConstantType::get(llvm_type, elements);
}

// For each instruction InstT that can be emitted as a constant, there is a
// function below to convert it to an `llvm::Constant*`:
//
// auto EmitAsConstant(ConstantContext& context, SemIR::InstT inst)
//     -> llvm::Constant*;

// Represent facet values the same as types.
static auto EmitAsConstant(ConstantContext& context, SemIR::FacetValue /*inst*/)
    -> llvm::Constant* {
  return context.GetTypeAsValue();
}

static auto EmitAsConstant(ConstantContext& context, SemIR::StructValue inst)
    -> llvm::Constant* {
  return EmitAggregateConstant<llvm::ConstantStruct>(
      context, inst.elements_id,
      cast<llvm::StructType>(context.GetType(inst.type_id)));
}

static auto EmitAsConstant(ConstantContext& context, SemIR::TupleValue inst)
    -> llvm::Constant* {
  // TODO: Add an ArrayValue instruction and stop using TupleValues to represent
  // array constants.
  if (context.sem_ir().types().Is<SemIR::ArrayType>(inst.type_id)) {
    return EmitAggregateConstant<llvm::ConstantArray>(
        context, inst.elements_id,
        cast<llvm::ArrayType>(context.GetType(inst.type_id)));
  }

  return EmitAggregateConstant<llvm::ConstantStruct>(
      context, inst.elements_id,
      cast<llvm::StructType>(context.GetType(inst.type_id)));
}

static auto EmitAsConstant(ConstantContext& /*context*/, SemIR::AddrOf /*inst*/)
    -> llvm::Constant* {
  // TODO: Constant lvalue support. For now we have no constant lvalues, so we
  // should never form a constant AddrOf.
  CARBON_FATAL("AddrOf constants not supported yet");
}

static auto EmitAsConstant(ConstantContext& context,
                           SemIR::AssociatedEntity inst) -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitAsConstant(ConstantContext& context, SemIR::BaseDecl inst)
    -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitAsConstant(ConstantContext& context, SemIR::BoolLiteral inst)
    -> llvm::Constant* {
  return llvm::ConstantInt::get(llvm::Type::getInt1Ty(context.llvm_context()),
                                inst.value.index);
}

static auto EmitAsConstant(ConstantContext& context, SemIR::BoundMethod inst)
    -> llvm::Constant* {
  // Propagate just the function; the object is separately provided to the
  // enclosing call as an implicit argument.
  return context.GetConstant(inst.function_decl_id);
}

static auto EmitAsConstant(ConstantContext& context,
                           SemIR::CompleteTypeWitness inst) -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitAsConstant(ConstantContext& context, SemIR::FieldDecl inst)
    -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitAsConstant(ConstantContext& context, SemIR::FloatLiteral inst)
    -> llvm::Constant* {
  const llvm::APFloat& value = context.sem_ir().floats().Get(inst.float_id);
  return llvm::ConstantFP::get(context.GetType(inst.type_id), value);
}

static auto EmitAsConstant(ConstantContext& context, SemIR::IntValue inst)
    -> llvm::Constant* {
  auto* type = context.GetType(inst.type_id);

  // IntLiteral is represented as an empty struct. All other integer types are
  // represented as an LLVM integer type.
  auto* int_type = llvm::dyn_cast<llvm::IntegerType>(type);
  if (!int_type) {
    auto* int_literal_value = context.GetIntLiteralAsValue();
    CARBON_CHECK(int_literal_value->getType() == type);
    return int_literal_value;
  }

  int bit_width = int_type->getBitWidth();
  auto val = context.sem_ir().ints().GetAtWidth(inst.int_id, bit_width);
  return llvm::ConstantInt::get(type, val);
}

static auto EmitAsConstant(ConstantContext& context, SemIR::Namespace inst)
    -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitAsConstant(ConstantContext& context,
                           SemIR::SpecificFunction inst) -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitAsConstant(ConstantContext& /*context*/,
                           SemIR::StringLiteral inst) -> llvm::Constant* {
  CARBON_FATAL("TODO: Add support: {0}", inst);
}

// Tries to emit an LLVM constant value for this constant instruction. Centrally
// handles some common cases and then dispatches to the relevant EmitAsConstant
// overload based on the type of the instruction for the remaining cases.
template <typename InstT>
static auto MaybeEmitAsConstant(ConstantContext& context, InstT inst)
    -> llvm::Constant* {
  if constexpr (InstT::Kind.constant_kind() == SemIR::InstConstantKind::Never ||
                InstT::Kind.constant_kind() ==
                    SemIR::InstConstantKind::SymbolicOnly) {
    CARBON_FATAL("Unexpected constant instruction kind {0}", inst);
  } else if constexpr (!InstT::Kind.is_lowered()) {
    // This instruction has a constant value, but that constant value will never
    // be used by lowering.
    return nullptr;
  } else if constexpr (InstT::Kind.is_type() == SemIR::InstIsType::Always) {
    // All types are lowered to the same value.
    return context.GetTypeAsValue();
  } else {
    return EmitAsConstant(context, inst);
  }
}

auto LowerConstants(FileContext& file_context,
                    llvm::MutableArrayRef<llvm::Constant*> constants) -> void {
  ConstantContext context(file_context, constants);
  // Lower each constant in InstId order. This guarantees we lower the
  // dependencies of a constant before we lower the constant itself.
  for (auto [inst_id, const_id] :
       file_context.sem_ir().constant_values().enumerate()) {
    if (!const_id.has_value() || !const_id.is_concrete()) {
      // We are only interested in lowering concrete constants.
      continue;
    }
    auto defining_inst_id =
        file_context.sem_ir().constant_values().GetInstId(const_id);
    if (defining_inst_id != inst_id) {
      // This isn't the instruction that defines the constant.
      continue;
    }

    auto inst = file_context.sem_ir().insts().Get(inst_id);
    if (inst.type_id().has_value() &&
        !file_context.sem_ir().types().IsComplete(inst.type_id())) {
      // If a constant doesn't have a complete type, that means we imported it
      // but didn't actually use it.
      continue;
    }
    llvm::Constant* value = nullptr;
    CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND(Name)                 \
  case CARBON_KIND(SemIR::Name const_inst): {         \
    value = MaybeEmitAsConstant(context, const_inst); \
    break;                                            \
  }
#include "toolchain/sem_ir/inst_kind.def"
    }

    constants[inst_id.index] = value;
    context.SetLastLoweredConstantIndex(inst_id.index);
  }
}

}  // namespace Carbon::Lower
