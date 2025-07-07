// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_

#include "clang/Basic/CodeGenOptions.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "toolchain/lower/context.h"
#include "toolchain/lower/specific_coalescer.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::Lower {

// Context and shared functionality for lowering within a SemIR file.
class FileContext {
 public:
  using LoweredConstantStore =
      FixedSizeValueStore<SemIR::InstId, llvm::Constant*>;

  explicit FileContext(Context& context, const SemIR::File& sem_ir,
                       const SemIR::InstNamer* inst_namer,
                       llvm::raw_ostream* vlog_stream);

  // Creates the Clang `CodeGenerator` to generate LLVM module from imported C++
  // code. Returns null when not importing C++.
  auto CreateCppCodeGenerator() -> std::unique_ptr<clang::CodeGenerator>;

  // Prepares to lower code in this IR, by precomputing needed LLVM types,
  // constants, declarations, etc. Should only be called once, before we lower
  // anything in this file.
  auto PrepareToLower() -> void;

  // Lowers all the definitions provided by the SemIR::File to LLVM IR.
  auto LowerDefinitions() -> void;

  // Perform final cleanup tasks once all lowering has been completed.
  auto Finalize() -> void;

  // Gets a callable's function. Returns nullptr for a builtin or a function we
  // have not lowered.
  auto GetFunction(SemIR::FunctionId function_id,
                   SemIR::SpecificId specific_id = SemIR::SpecificId::None)
      -> llvm::Function* {
    return *GetFunctionAddr(function_id, specific_id);
  }

  // Gets a or creates callable's function. Returns nullptr for a builtin.
  auto GetOrCreateFunction(SemIR::FunctionId function_id,
                           SemIR::SpecificId specific_id) -> llvm::Function*;

  // Returns a lowered type for the given type_id.
  auto GetType(SemIR::TypeId type_id) -> llvm::Type* {
    CARBON_CHECK(type_id.has_value(), "Should not be called with `None`");
    CARBON_CHECK(type_id.is_concrete(), "Lowering symbolic type {0}: {1}",
                 type_id, sem_ir().types().GetAsInst(type_id));
    CARBON_CHECK(types_.Get(type_id), "Missing type {0}: {1}", type_id,
                 sem_ir().types().GetAsInst(type_id));
    return types_.Get(type_id);
  }

  // Returns location information for use with DebugInfo.
  auto GetLocForDI(SemIR::InstId inst_id) -> Context::LocForDI;

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Constant* {
    return context().GetTypeAsValue();
  }

  // Returns a lowered value to use for a value of int literal type.
  auto GetIntLiteralAsValue() -> llvm::Constant* {
    return context().GetIntLiteralAsValue();
  }

  // Returns a value for the given constant. If specified, `use_inst_id` is the
  // instruction that is using this constant.
  auto GetConstant(SemIR::ConstantId const_id, SemIR::InstId use_inst_id)
      -> llvm::Value*;

  auto GetVtable(SemIR::VtableId vtable_id, SemIR::SpecificId specific_id)
      -> llvm::GlobalVariable* {
    if (!specific_id.has_value()) {
      return vtables_.Get(vtable_id);
    }
    auto*& specific_vtable = specific_vtables_.Get(specific_id);
    if (!specific_vtable) {
      specific_vtable =
          BuildVtable(sem_ir().vtables().Get(vtable_id), specific_id);
    }
    return specific_vtable;
  }

  // Returns the empty LLVM struct type used to represent the type `type`.
  auto GetTypeType() -> llvm::StructType* { return context().GetTypeType(); }

  auto context() -> Context& { return *context_; }
  auto llvm_context() -> llvm::LLVMContext& { return context().llvm_context(); }
  auto llvm_module() -> llvm::Module& { return context().llvm_module(); }
  auto cpp_code_generator() -> clang::CodeGenerator& {
    CARBON_CHECK(cpp_code_generator_);
    return *cpp_code_generator_;
  }
  auto sem_ir() const -> const SemIR::File& { return *sem_ir_; }
  auto cpp_ast() -> const clang::ASTUnit* { return sem_ir().cpp_ast(); }
  auto inst_namer() -> const SemIR::InstNamer* { return inst_namer_; }
  auto global_variables() -> const Map<SemIR::InstId, llvm::GlobalVariable*>& {
    return global_variables_;
  }
  auto printf_int_format_string() -> llvm::Value* {
    return context().printf_int_format_string();
  }
  auto SetPrintfIntFormatString(llvm::Value* printf_int_format_string) {
    context().SetPrintfIntFormatString(printf_int_format_string);
  }

  struct FunctionTypeInfo {
    llvm::FunctionType* type;
    llvm::SmallVector<SemIR::InstId> param_inst_ids;
    llvm::Type* return_type = nullptr;
    SemIR::InstId return_param_id = SemIR::InstId::None;
  };

  // Retrieve various features of the function's type useful for constructing
  // the `llvm::Type` for the `llvm::Function`. If any part of the type can't be
  // manifest (eg: incomplete return or parameter types), then the result is as
  // if the type was `void()`.
  auto BuildFunctionTypeInfo(const SemIR::Function& function,
                             SemIR::SpecificId specific_id) -> FunctionTypeInfo;

  // Builds the global for the given instruction, which should then be cached by
  // the caller.
  auto BuildGlobalVariableDecl(SemIR::VarStorage var_storage)
      -> llvm::GlobalVariable*;

  // Builds the definition for the given function. If the function is only a
  // declaration with no definition, does nothing. If this is a generic it'll
  // only be lowered if the specific_id is specified. During this lowering of
  // a generic, more generic functions may be added for lowering.
  auto BuildFunctionDefinition(
      SemIR::FunctionId function_id,
      SemIR::SpecificId specific_id = SemIR::SpecificId::None) -> void;

 private:
  // Gets the location in which a callable's function is stored.
  auto GetFunctionAddr(SemIR::FunctionId function_id,
                       SemIR::SpecificId specific_id) -> llvm::Function** {
    return specific_id.has_value() ? &specific_functions_.Get(specific_id)
                                   : &functions_.Get(function_id);
  }

  // Notes that a C++ function has been referenced for the first time, so we
  // should ask Clang to generate a definition for it if possible.
  auto HandleReferencedCppFunction(clang::FunctionDecl* cpp_decl) -> void;

  // Notes that a specific function has been referenced for the first time.
  // Updates the fingerprint to include the function's type, and adds the
  // function to the list of specific functions whose definitions should be
  // lowered.
  auto HandleReferencedSpecificFunction(SemIR::FunctionId function_id,
                                        SemIR::SpecificId specific_id,
                                        llvm::Type* llvm_type) -> void;

  // Builds the declaration for the given function, which should then be cached
  // by the caller.
  auto BuildFunctionDecl(SemIR::FunctionId function_id,
                         SemIR::SpecificId specific_id =
                             SemIR::SpecificId::None) -> llvm::Function*;

  // Builds a function's body. Common functionality for all functions.
  //
  // The `function_id` and `specific_id` identify the function within this
  // context's file. If the function was defined in a different file,
  // `definition_context` is a `FileContext` for that other file.
  // `definition_function` is the `Function` object within the file that owns
  // the definition.
  auto BuildFunctionBody(SemIR::FunctionId function_id,
                         SemIR::SpecificId specific_id,
                         const SemIR::Function& declaration_function,
                         FileContext& definition_context,
                         const SemIR::Function& definition_function) -> void;

  // Build the DISubprogram metadata for the given function.
  auto BuildDISubprogram(const SemIR::Function& function,
                         const llvm::Function* llvm_function)
      -> llvm::DISubprogram*;

  // Builds the type for the given instruction, which should then be cached by
  // the caller.
  auto BuildType(SemIR::InstId inst_id) -> llvm::Type*;

  auto BuildVtable(const SemIR::Vtable& vtable, SemIR::SpecificId specific_id)
      -> llvm::GlobalVariable*;

  // Records a specific that was lowered for a generic. These are added one
  // by one while lowering their definitions.
  auto AddLoweredSpecificForGeneric(SemIR::GenericId generic_id,
                                    SemIR::SpecificId specific_id) {
    lowered_specifics_.Get(generic_id).push_back(specific_id);
  }

  // The overall lowering context.
  Context* context_;

  // The input SemIR.
  const SemIR::File* const sem_ir_;

  // The options used to create the Clang Code Generator.
  clang::HeaderSearchOptions cpp_header_search_options_;
  clang::PreprocessorOptions cpp_preprocessor_options_;
  clang::CodeGenOptions cpp_code_gen_options_;

  // The Clang `CodeGenerator` to generate LLVM module from imported C++
  // code. Should be initialized using `CreateCppCodeGenerator()`. Can be null
  // if no C++ code is imported.
  std::unique_ptr<clang::CodeGenerator> cpp_code_generator_;

  // The instruction namer, if given.
  const SemIR::InstNamer* const inst_namer_;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // Maps callables to lowered functions. SemIR treats callables as the
  // canonical form of a function, so lowering needs to do the same.
  using LoweredFunctionStore =
      FixedSizeValueStore<SemIR::FunctionId, llvm::Function*>;
  LoweredFunctionStore functions_;

  // Maps specific callables to lowered functions.
  FixedSizeValueStore<SemIR::SpecificId, llvm::Function*> specific_functions_;

  // Provides lowered versions of types. Entries are non-symbolic types.
  using LoweredTypeStore = FixedSizeValueStore<SemIR::TypeId, llvm::Type*>;
  LoweredTypeStore types_;

  // Maps constants to their lowered values. Indexes are the `InstId` for
  // constant instructions.
  LoweredConstantStore constants_;

  // Maps global variables to their lowered variant.
  Map<SemIR::InstId, llvm::GlobalVariable*> global_variables_;

  // For a generic function, keep track of the specifics for which LLVM
  // function declarations were created. Those can be retrieved then from
  // `specific_functions_`.
  FixedSizeValueStore<SemIR::GenericId, llvm::SmallVector<SemIR::SpecificId>>
      lowered_specifics_;

  SpecificCoalescer coalescer_;

  FixedSizeValueStore<SemIR::VtableId, llvm::GlobalVariable*> vtables_;
  FixedSizeValueStore<SemIR::SpecificId, llvm::GlobalVariable*>
      specific_vtables_;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
