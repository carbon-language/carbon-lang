// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Value.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Lower {

// Context and shared functionality for lowering constant values.
class ConstantContext {
 public:
  explicit ConstantContext(FileContext& file_context,
                           llvm::MutableArrayRef<llvm::Constant*> constants)
      : file_context_(&file_context), constants_(constants) {}

  // Gets the lowered constant value for an instruction, which must have a
  // constant value that has already been lowered.
  auto GetConstant(SemIR::InstId inst_id) const -> llvm::Constant* {
    return GetConstant(file_context_->sem_ir().constant_values().Get(inst_id));
  }

  // Gets the lowered constant value for a constant that has already been
  // lowered.
  auto GetConstant(SemIR::ConstantId const_id) const -> llvm::Constant* {
    CARBON_CHECK(const_id.is_template() && const_id.inst_id().index >= 0)
        << "Unexpected constant ID " << const_id;
    CARBON_CHECK(const_id.inst_id().index <= last_lowered_constant_index_)
        << "Queried constant " << const_id << " that has not been lowered yet";
    return constants_[const_id.inst_id().index];
  }

  // Returns a constant for the case of a value that should never be used.
  auto GetUnusedConstant(SemIR::TypeId /*type_id*/) const -> llvm::Constant* {
    // TODO: Consider using a poison value of the appropriate type.
    return nullptr;
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

// For each instruction kind that can produce a constant, there is a function
// below to convert it to an `llvm::Constant*`:
//
// auto Emit<InstKind>AsConstant(ConstantContext& context,
//                               SemIR::<InstKind> inst) -> llvm::Constant*;

// For constants that are always of type `type`, produce the trivial runtime
// representation of type `type`.
#define CARBON_SEM_IR_INST_KIND_TYPE_NEVER(...)
#define CARBON_SEM_IR_INST_KIND_TYPE_MAYBE(...)
#define CARBON_SEM_IR_INST_KIND_CONSTANT_SYMBOLIC_ONLY(...)
#define CARBON_SEM_IR_INST_KIND(Name)                                      \
  static auto Emit##Name##AsConstant(                                      \
      ConstantContext& context, SemIR::Name /*inst*/) -> llvm::Constant* { \
    return context.GetTypeAsValue();                                       \
  }
#include "toolchain/sem_ir/inst_kind.def"

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

static auto EmitStructValueAsConstant(ConstantContext& context,
                                      SemIR::StructValue inst)
    -> llvm::Constant* {
  return EmitAggregateConstant<llvm::ConstantStruct>(
      context, inst.elements_id,
      cast<llvm::StructType>(context.GetType(inst.type_id)));
}

static auto EmitTupleValueAsConstant(ConstantContext& context,
                                     SemIR::TupleValue inst)
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

static auto EmitAddrOfAsConstant(ConstantContext& /*context*/,
                                 SemIR::AddrOf /*inst*/) -> llvm::Constant* {
  // TODO: Constant lvalue support. For now we have no constant lvalues, so we
  // should never form a constant AddrOf.
  CARBON_FATAL() << "AddrOf constants not supported yet";
}

static auto EmitAssociatedEntityAsConstant(ConstantContext& context,
                                           SemIR::AssociatedEntity inst)
    -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitBaseDeclAsConstant(ConstantContext& context,
                                   SemIR::BaseDecl inst) -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitBoolLiteralAsConstant(ConstantContext& context,
                                      SemIR::BoolLiteral inst)
    -> llvm::Constant* {
  return llvm::ConstantInt::get(llvm::Type::getInt1Ty(context.llvm_context()),
                                inst.value.index);
}

static auto EmitBoundMethodAsConstant(ConstantContext& context,
                                      SemIR::BoundMethod inst)
    -> llvm::Constant* {
  // Propagate just the function; the object is separately provided to the
  // enclosing call as an implicit argument.
  return context.GetConstant(inst.function_id);
}

static auto EmitFieldDeclAsConstant(ConstantContext& context,
                                    SemIR::FieldDecl inst) -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitFloatLiteralAsConstant(ConstantContext& context,
                                       SemIR::FloatLiteral inst)
    -> llvm::Constant* {
  const llvm::APFloat& value = context.sem_ir().floats().Get(inst.float_id);
  return llvm::ConstantFP::get(context.GetType(inst.type_id), value);
}

static auto EmitInterfaceWitnessAsConstant(ConstantContext& context,
                                           SemIR::InterfaceWitness inst)
    -> llvm::Constant* {
  // TODO: For dynamic dispatch, we might want to lower witness tables as
  // constants.
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitIntLiteralAsConstant(ConstantContext& context,
                                     SemIR::IntLiteral inst)
    -> llvm::Constant* {
  return llvm::ConstantInt::get(context.GetType(inst.type_id),
                                context.sem_ir().ints().Get(inst.int_id));
}

static auto EmitNamespaceAsConstant(ConstantContext& context,
                                    SemIR::Namespace inst) -> llvm::Constant* {
  return context.GetUnusedConstant(inst.type_id);
}

static auto EmitRealLiteralAsConstant(ConstantContext& context,
                                      SemIR::RealLiteral inst)
    -> llvm::Constant* {
  const Real& real = context.sem_ir().reals().Get(inst.real_id);
  // TODO: This will probably have overflow issues, and should be fixed.
  double val =
      real.mantissa.getZExtValue() *
      std::pow((real.is_decimal ? 10 : 2), real.exponent.getSExtValue());
  llvm::APFloat llvm_val(val);
  return llvm::ConstantFP::get(context.GetType(inst.type_id), llvm_val);
}

static auto EmitStringLiteralAsConstant(ConstantContext& /*context*/,
                                        SemIR::StringLiteral inst)
    -> llvm::Constant* {
  CARBON_FATAL() << "TODO: Add support: " << inst;
}

static auto EmitStructTypeFieldAsConstant(ConstantContext& /*context*/,
                                          SemIR::StructTypeField /*inst*/)
    -> llvm::Constant* {
  // A StructTypeField isn't a value, so this constant value won't ever be used.
  // It also doesn't even have a type, so we can't use GetUnusedConstant.
  return nullptr;
}

auto LowerConstants(FileContext& file_context,
                    llvm::MutableArrayRef<llvm::Constant*> constants) -> void {
  ConstantContext context(file_context, constants);
  // Lower each constant in InstId order. This guarantees we lower the
  // dependencies of a constant before we lower the constant itself.
  for (auto [inst_id_val, const_id] :
       llvm::enumerate(file_context.sem_ir().constant_values().array_ref())) {
    if (!const_id.is_valid() || !const_id.is_template()) {
      // We are only interested in lowering template constants.
      continue;
    }

    if (const_id.inst_id().index != static_cast<int32_t>(inst_id_val)) {
      // This isn't the instruction that defines the constant.
      continue;
    }

    auto inst = file_context.sem_ir().insts().Get(const_id.inst_id());
    llvm::Constant* value = nullptr;
    CARBON_KIND_SWITCH(inst) {
#define CARBON_SEM_IR_INST_KIND_CONSTANT_NEVER(...)
#define CARBON_SEM_IR_INST_KIND_CONSTANT_SYMBOLIC_ONLY(...)
#define CARBON_SEM_IR_INST_KIND(Name)                    \
  case CARBON_KIND(SemIR::Name const_inst):              \
    value = Emit##Name##AsConstant(context, const_inst); \
    break;
#include "toolchain/sem_ir/inst_kind.def"

      default:
        CARBON_FATAL() << "Unexpected constant instruction kind " << inst;
    }

    constants[const_id.inst_id().index] = value;
    context.SetLastLoweredConstantIndex(const_id.inst_id().index);
  }
}

}  // namespace Carbon::Lower
