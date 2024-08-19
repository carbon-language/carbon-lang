// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_

#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::Lower {

// Context and shared functionality for lowering handlers.
class FileContext {
 public:
  explicit FileContext(llvm::LLVMContext& llvm_context, bool include_debug_info,
                       llvm::StringRef module_name, const SemIR::File& sem_ir,
                       const SemIR::InstNamer* inst_namer,
                       llvm::raw_ostream* vlog_stream);

  // Lowers the SemIR::File to LLVM IR. Should only be called once, and handles
  // the main execution loop.
  auto Run() -> std::unique_ptr<llvm::Module>;

  // Create the DICompileUnit metadata for this compilation.
  auto BuildDICompileUnit(llvm::StringRef module_name) -> void;

  // Gets a callable's function. Returns nullptr for a builtin.
  auto GetFunction(SemIR::FunctionId function_id) -> llvm::Function* {
    return functions_[function_id.index];
  }

  // Returns a lowered type for the given type_id.
  auto GetType(SemIR::TypeId type_id) -> llvm::Type* {
    // InvalidType should not be passed in.
    CARBON_CHECK(type_id.index >= 0) << type_id;
    CARBON_CHECK(types_[type_id.index]) << "Missing type " << type_id;
    return types_[type_id.index];
  }

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Constant* {
    return llvm::ConstantStruct::get(GetTypeType());
  }

  // Returns a global value for the given instruction.
  auto GetGlobal(SemIR::InstId inst_id) -> llvm::Value*;

  // Returns the empty LLVM struct type used to represent the type `type`.
  auto GetTypeType() -> llvm::StructType* {
    if (!type_type_) {
      // `type` is lowered to an empty LLVM StructType.
      type_type_ = llvm::StructType::create(*llvm_context_, {}, "type");
    }
    return type_type_;
  }

  auto llvm_context() -> llvm::LLVMContext& { return *llvm_context_; }
  auto llvm_module() -> llvm::Module& { return *llvm_module_; }
  auto sem_ir() -> const SemIR::File& { return *sem_ir_; }
  auto inst_namer() -> const SemIR::InstNamer* { return inst_namer_; }

 private:
  // Builds the declaration for the given function, which should then be cached
  // by the caller.
  auto BuildFunctionDecl(SemIR::FunctionId function_id) -> llvm::Function*;

  // Builds the definition for the given function. If the function is only a
  // declaration with no definition, does nothing.
  auto BuildFunctionDefinition(SemIR::FunctionId function_id) -> void;

  // Builds the type for the given instruction, which should then be cached by
  // the caller.
  auto BuildType(SemIR::InstId inst_id) -> llvm::Type*;

  // State for building the LLVM IR.
  llvm::LLVMContext* llvm_context_;
  std::unique_ptr<llvm::Module> llvm_module_;
  llvm::DIBuilder di_builder_;
  llvm::DICompileUnit* di_compile_unit_;

  // The input SemIR.
  const SemIR::File* const sem_ir_;

  // The instruction namer, if given.
  const SemIR::InstNamer* const inst_namer_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // Maps callables to lowered functions. SemIR treats callables as the
  // canonical form of a function, so lowering needs to do the same.
  // We resize this directly to the (often large) correct size.
  llvm::SmallVector<llvm::Function*, 0> functions_;

  // Provides lowered versions of types.
  // We resize this directly to the (often large) correct size.
  llvm::SmallVector<llvm::Type*, 0> types_;

  // Lowered version of the builtin type `type`.
  llvm::StructType* type_type_ = nullptr;

  // Maps constants to their lowered values.
  // We resize this directly to the (often large) correct size.
  llvm::SmallVector<llvm::Constant*, 0> constants_;

  // Specify whether to include debug info metadata in the generated LLVM IR.
  bool include_debug_info_;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
