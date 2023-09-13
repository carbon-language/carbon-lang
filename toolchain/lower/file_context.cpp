// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/file_context.h"

#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/node.h"
#include "toolchain/sem_ir/node_kind.h"

namespace Carbon::Lower {

FileContext::FileContext(llvm::LLVMContext& llvm_context,
                         llvm::StringRef module_name,
                         const SemIR::File& semantics_ir,
                         llvm::raw_ostream* vlog_stream)
    : llvm_context_(&llvm_context),
      llvm_module_(std::make_unique<llvm::Module>(module_name, llvm_context)),
      semantics_ir_(&semantics_ir),
      vlog_stream_(vlog_stream) {
  CARBON_CHECK(!semantics_ir.has_errors())
      << "Generating LLVM IR from invalid SemIR::File is unsupported.";
}

// TODO: Move this to lower.cpp.
auto FileContext::Run() -> std::unique_ptr<llvm::Module> {
  CARBON_CHECK(llvm_module_) << "Run can only be called once.";

  // Lower types.
  auto types = semantics_ir_->types();
  types_.resize_for_overwrite(types.size());
  for (auto [i, type] : llvm::enumerate(types)) {
    types_[i] = BuildType(type);
  }

  // Lower function declarations.
  functions_.resize_for_overwrite(semantics_ir_->functions_size());
  for (auto i : llvm::seq(semantics_ir_->functions_size())) {
    functions_[i] = BuildFunctionDeclaration(SemIR::FunctionId(i));
  }

  // TODO: Lower global variable declarations.

  // Lower function definitions.
  for (auto i : llvm::seq(semantics_ir_->functions_size())) {
    BuildFunctionDefinition(SemIR::FunctionId(i));
  }

  // TODO: Lower global variable initializers.

  return std::move(llvm_module_);
}

auto FileContext::BuildFunctionDeclaration(SemIR::FunctionId function_id)
    -> llvm::Function* {
  const auto& function = semantics_ir().GetFunction(function_id);
  const bool has_return_slot = function.return_slot_id.is_valid();
  auto param_refs = semantics_ir().GetNodeBlock(function.param_refs_id);

  SemIR::InitializingRepresentation return_rep =
      function.return_type_id.is_valid()
          ? SemIR::GetInitializingRepresentation(semantics_ir(),
                                                 function.return_type_id)
          : SemIR::InitializingRepresentation{
                .kind = SemIR::InitializingRepresentation::None};
  CARBON_CHECK(return_rep.has_return_slot() == has_return_slot);

  llvm::SmallVector<llvm::Type*> param_types;
  // TODO: Consider either storing `param_node_ids` somewhere so that we can
  // reuse it from `BuildFunctionDefinition` and when building calls, or factor
  // out a mechanism to compute the mapping between parameters and arguments on
  // demand.
  llvm::SmallVector<SemIR::NodeId> param_node_ids;
  param_types.reserve(has_return_slot + param_refs.size());
  param_node_ids.reserve(has_return_slot + param_refs.size());
  if (has_return_slot) {
    param_types.push_back(GetType(function.return_type_id)->getPointerTo());
    param_node_ids.push_back(function.return_slot_id);
  }
  for (auto param_ref_id : param_refs) {
    auto param_type_id = semantics_ir().GetNode(param_ref_id).type_id();
    switch (auto value_rep =
                SemIR::GetValueRepresentation(semantics_ir(), param_type_id);
            value_rep.kind) {
      case SemIR::ValueRepresentation::None:
        break;
      case SemIR::ValueRepresentation::Copy:
      case SemIR::ValueRepresentation::Custom:
        param_types.push_back(GetType(value_rep.type));
        param_node_ids.push_back(param_ref_id);
        break;
      case SemIR::ValueRepresentation::Pointer:
        param_types.push_back(GetType(value_rep.type)->getPointerTo());
        param_node_ids.push_back(param_ref_id);
        break;
    }
  }

  // If the initializing representation doesn't produce a value, set the return
  // type to void.
  llvm::Type* return_type =
      return_rep.kind == SemIR::InitializingRepresentation::ByCopy
          ? GetType(function.return_type_id)
          : llvm::Type::getVoidTy(llvm_context());

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(return_type, param_types, /*isVarArg=*/false);
  auto* llvm_function = llvm::Function::Create(
      function_type, llvm::Function::ExternalLinkage,
      semantics_ir().GetString(function.name_id), llvm_module());

  // Set up parameters and the return slot.
  for (auto [node_id, arg] :
       llvm::zip_equal(param_node_ids, llvm_function->args())) {
    if (node_id == function.return_slot_id) {
      arg.setName("return");
      arg.addAttr(llvm::Attribute::getWithStructRetType(
          llvm_context(), GetType(function.return_type_id)));
    } else {
      arg.setName(semantics_ir().GetString(
          semantics_ir().GetNode(node_id).GetAsParameter()));
    }
  }

  return llvm_function;
}

auto FileContext::BuildFunctionDefinition(SemIR::FunctionId function_id)
    -> void {
  const auto& function = semantics_ir().GetFunction(function_id);
  const auto& body_block_ids = function.body_block_ids;
  if (body_block_ids.empty()) {
    // Function is probably defined in another file; not an error.
    return;
  }

  llvm::Function* llvm_function = GetFunction(function_id);
  FunctionContext function_lowering(*this, llvm_function);

  const bool has_return_slot = function.return_slot_id.is_valid();

  // Add parameters to locals.
  // TODO: This duplicates the mapping between semantics nodes and LLVM
  // function parameters that was already computed in BuildFunctionDeclaration.
  // We should only do that once.
  auto param_refs = semantics_ir().GetNodeBlock(function.param_refs_id);
  int param_index = 0;
  if (has_return_slot) {
    function_lowering.SetLocal(function.return_slot_id,
                               llvm_function->getArg(param_index));
    ++param_index;
  }
  for (auto param_ref_id : param_refs) {
    auto param_type_id = semantics_ir().GetNode(param_ref_id).type_id();
    if (SemIR::GetValueRepresentation(semantics_ir(), param_type_id).kind ==
        SemIR::ValueRepresentation::None) {
      function_lowering.SetLocal(
          param_ref_id, llvm::PoisonValue::get(GetType(param_type_id)));
    } else {
      function_lowering.SetLocal(param_ref_id,
                                 llvm_function->getArg(param_index));
      ++param_index;
    }
  }

  // Lower all blocks.
  for (auto block_id : body_block_ids) {
    CARBON_VLOG() << "Lowering " << block_id << "\n";
    auto* llvm_block = function_lowering.GetBlock(block_id);
    // Keep the LLVM blocks in lexical order.
    llvm_block->moveBefore(llvm_function->end());
    function_lowering.builder().SetInsertPoint(llvm_block);
    for (const auto& node_id : semantics_ir().GetNodeBlock(block_id)) {
      auto node = semantics_ir().GetNode(node_id);
      CARBON_VLOG() << "Lowering " << node_id << ": " << node << "\n";
      // clang warns on unhandled enum values; clang-tidy is incorrect here.
      // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
      switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)            \
  case SemIR::NodeKind::Name:                       \
    Handle##Name(function_lowering, node_id, node); \
    break;
#include "toolchain/sem_ir/node_kind.def"
      }
    }
  }
}

auto FileContext::BuildType(SemIR::NodeId node_id) -> llvm::Type* {
  switch (node_id.index) {
    case SemIR::BuiltinKind::FloatingPointType.AsInt():
      // TODO: Handle different sizes.
      return llvm::Type::getDoubleTy(*llvm_context_);
    case SemIR::BuiltinKind::IntegerType.AsInt():
      // TODO: Handle different sizes.
      return llvm::Type::getInt32Ty(*llvm_context_);
    case SemIR::BuiltinKind::BoolType.AsInt():
      // TODO: We may want to have different representations for `bool` storage
      // (`i8`) versus for `bool` values (`i1`).
      return llvm::Type::getInt1Ty(*llvm_context_);
    default:
      // Handled below.
      break;
  }

  auto node = semantics_ir_->GetNode(node_id);
  switch (node.kind()) {
    case SemIR::NodeKind::ArrayType: {
      auto [bound_node_id, type_id] = node.GetAsArrayType();
      return llvm::ArrayType::get(
          GetType(type_id), semantics_ir_->GetArrayBoundValue(bound_node_id));
    }
    case SemIR::NodeKind::ConstType:
      return GetType(node.GetAsConstType());
    case SemIR::NodeKind::PointerType:
      return llvm::PointerType::get(*llvm_context_, /*AddressSpace=*/0);
    case SemIR::NodeKind::StructType: {
      auto refs = semantics_ir_->GetNodeBlock(node.GetAsStructType());
      llvm::SmallVector<llvm::Type*> subtypes;
      subtypes.reserve(refs.size());
      for (auto ref_id : refs) {
        auto [field_name_id, field_type_id] =
            semantics_ir_->GetNode(ref_id).GetAsStructTypeField();
        // TODO: Handle recursive types. The restriction for builtins prevents
        // recursion while still letting them cache.
        CARBON_CHECK(field_type_id.index < SemIR::BuiltinKind::ValidCount)
            << field_type_id;
        subtypes.push_back(GetType(field_type_id));
      }
      return llvm::StructType::get(*llvm_context_, subtypes);
    }
    case SemIR::NodeKind::TupleType: {
      // TODO: Investigate special-casing handling of empty tuples so that they
      // can be collectively replaced with LLVM's void, particularly around
      // function returns. LLVM doesn't allow declaring variables with a void
      // type, so that may require significant special casing.
      auto refs = semantics_ir_->GetTypeBlock(node.GetAsTupleType());
      llvm::SmallVector<llvm::Type*> subtypes;
      subtypes.reserve(refs.size());
      for (auto ref_id : refs) {
        subtypes.push_back(GetType(ref_id));
      }
      return llvm::StructType::get(*llvm_context_, subtypes);
    }
    default: {
      CARBON_FATAL() << "Cannot use node as type: " << node_id;
    }
  }
}

}  // namespace Carbon::Lower
