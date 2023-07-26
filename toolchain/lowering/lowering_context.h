// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWERING_LOWERING_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWERING_LOWERING_CONTEXT_H_

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_node.h"

namespace Carbon {

// Context and shared functionality for lowering handlers.
class LoweringContext {
 public:
  explicit LoweringContext(llvm::LLVMContext& llvm_context,
                           llvm::StringRef module_name,
                           const SemanticsIR& semantics_ir,
                           llvm::raw_ostream* vlog_stream);

  // Lowers the SemanticsIR to LLVM IR. Should only be called once, and handles
  // the main execution loop.
  auto Run() -> std::unique_ptr<llvm::Module>;

  // Gets a callable's function.
  auto GetFunction(SemanticsFunctionId function_id) -> llvm::Function* {
    CARBON_CHECK(functions_[function_id.index] != nullptr) << function_id;
    return functions_[function_id.index];
  }

  // Returns a lowered type for the given type_id.
  auto GetType(SemanticsTypeId type_id) -> llvm::Type* {
    // InvalidType should not be passed in.
    if (type_id == SemanticsTypeId::TypeType) {
      // `type` is lowered to an empty LLVM StructType.
      return llvm::StructType::get(llvm_context());
    }
    // Function with no return type gets void type.
    if (type_id == SemanticsTypeId::Invalid) {
      return llvm::Type::getVoidTy(llvm_context());
    }
    CARBON_CHECK(type_id.index >= 0) << type_id;
    return types_[type_id.index];
  }

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Value* {
    return llvm::ConstantStruct::getAnon(llvm_context(), {});
  }

  auto llvm_context() -> llvm::LLVMContext& { return *llvm_context_; }
  auto llvm_module() -> llvm::Module& { return *llvm_module_; }
  auto semantics_ir() -> const SemanticsIR& { return *semantics_ir_; }

 private:
  // Builds the declaration for the given function, which should then be cached
  // by the caller.
  auto BuildFunctionDeclaration(SemanticsFunctionId function_id)
      -> llvm::Function*;

  // Builds the definition for the given function. If the function is only a
  // declaration with no definition, does nothing.
  auto BuildFunctionDefinition(SemanticsFunctionId function_id) -> void;

  // Builds the type for the given node, which should then be cached by the
  // caller.
  auto BuildType(SemanticsNodeId node_id) -> llvm::Type*;

  // State for building the LLVM IR.
  llvm::LLVMContext* llvm_context_;
  std::unique_ptr<llvm::Module> llvm_module_;

  // The input Semantics IR.
  const SemanticsIR* const semantics_ir_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // Maps callables to lowered functions. Semantics treats callables as the
  // canonical form of a function, so lowering needs to do the same.
  llvm::SmallVector<llvm::Function*> functions_;

  // Provides lowered versions of types.
  llvm::SmallVector<llvm::Type*> types_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_LOWERING_LOWERING_CONTEXT_H_
