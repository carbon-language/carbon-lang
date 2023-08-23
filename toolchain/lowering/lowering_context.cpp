// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lowering/lowering_context.h"

#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/lowering/lowering_function_context.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"
#include "toolchain/semantics/semantics_node_kind.h"

namespace Carbon {

LoweringContext::LoweringContext(llvm::LLVMContext& llvm_context,
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

// TODO: Move this to lower_to_llvm.cpp.
auto LoweringContext::Run() -> std::unique_ptr<llvm::Module> {
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

auto LoweringContext::BuildFunctionDeclaration(SemIR::FunctionId function_id)
    -> llvm::Function* {
  auto function = semantics_ir().GetFunction(function_id);

  // TODO: Lower type information for the arguments prior to building args.
  auto param_refs = semantics_ir().GetNodeBlock(function.param_refs_id);
  llvm::SmallVector<llvm::Type*> args;
  args.resize_for_overwrite(param_refs.size());
  for (auto [i, param_ref] : llvm::enumerate(param_refs)) {
    args[i] = GetType(semantics_ir().GetNode(param_ref).type_id());
  }

  // If return type is not valid, the function does not have a return type.
  // Hence, set return type to void.
  llvm::Type* return_type = function.return_type_id.is_valid()
                                ? GetType(function.return_type_id)
                                : llvm::Type::getVoidTy(llvm_context());
  llvm::FunctionType* function_type =
      llvm::FunctionType::get(return_type, args, /*isVarArg=*/false);
  auto* llvm_function = llvm::Function::Create(
      function_type, llvm::Function::ExternalLinkage,
      semantics_ir().GetString(function.name_id), llvm_module());

  // Set parameter names.
  for (auto [i, param_ref] : llvm::enumerate(param_refs)) {
    auto name_id = semantics_ir().GetNode(param_ref).GetAsParameter();
    llvm_function->getArg(i)->setName(semantics_ir().GetString(name_id));
  }

  return llvm_function;
}

auto LoweringContext::BuildFunctionDefinition(SemIR::FunctionId function_id)
    -> void {
  auto function = semantics_ir().GetFunction(function_id);
  const auto& body_block_ids = function.body_block_ids;
  if (body_block_ids.empty()) {
    // Function is probably defined in another file; not an error.
    return;
  }

  llvm::Function* llvm_function = GetFunction(function_id);
  LoweringFunctionContext function_lowering(*this, llvm_function);

  // Add parameters to locals.
  auto param_refs = semantics_ir().GetNodeBlock(function.param_refs_id);
  for (auto [i, param_ref] : llvm::enumerate(param_refs)) {
    function_lowering.SetLocal(param_ref, llvm_function->getArg(i));
  }

  // Lower all blocks.
  // TODO: Determine the set of reachable blocks, and only lower those ones.
  for (auto block_id : body_block_ids) {
    CARBON_VLOG() << "Lowering " << block_id << "\n";
    auto* llvm_block = function_lowering.GetBlock(block_id);
    // Keep the LLVM blocks in lexical order.
    llvm_block->moveBefore(llvm_function->end());
    function_lowering.builder().SetInsertPoint(llvm_block);
    for (const auto& node_id : semantics_ir().GetNodeBlock(block_id)) {
      auto node = semantics_ir().GetNode(node_id);
      CARBON_VLOG() << "Lowering " << node_id << ": " << node << "\n";
      switch (node.kind()) {
#define CARBON_SEMANTICS_NODE_KIND(Name)                    \
  case SemIR::NodeKind::Name:                               \
    LoweringHandle##Name(function_lowering, node_id, node); \
    break;
#include "toolchain/semantics/semantics_node_kind.def"
      }
    }
  }
}

auto LoweringContext::BuildType(SemIR::NodeId node_id) -> llvm::Type* {
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

}  // namespace Carbon
