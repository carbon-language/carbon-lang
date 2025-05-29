// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_

#include "clang/Basic/CodeGenOptions.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "common/raw_string_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/BLAKE3.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_namer.h"

namespace Carbon::Lower {

// Context and shared functionality for lowering handlers.
class FileContext {
 public:
  // Location information for use with DebugInfo. The line_number and
  // column_number are >= 0, with 0 as unknown, so that they can be passed
  // directly to DebugInfo.
  struct LocForDI {
    llvm::StringRef filename;
    int32_t line_number;
    int32_t column_number;
  };

  // Describes a specific function's body fingerprint.
  struct SpecificFunctionFingerprint {
    // Fingerprint with all specific-dependent instructions, except specific
    // calls. This is built by the `FunctionContext` while lowering each
    // instruction in the definition of a specific function.
    // TODO: This can be merged with the function type fingerprint, for a
    // single upfront non-equivalence check, and hash bucketing for deeper
    // equivalence evaluation.
    llvm::BLAKE3Result<32> common_fingerprint;
    // Fingerprint for all calls to specific functions (hashes all calls to
    // other specifics). This is built by the `FunctionContext` while lowering.
    llvm::BLAKE3Result<32> specific_fingerprint;
    // All non-hashed specific_ids of functions called.
    llvm::SmallVector<SemIR::SpecificId> calls;
  };

  explicit FileContext(
      llvm::LLVMContext& llvm_context,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
      std::optional<llvm::ArrayRef<Parse::GetTreeAndSubtreesFn>>
          tree_and_subtrees_getters_for_debug_info,
      llvm::StringRef module_name, const SemIR::File& sem_ir,
      clang::ASTUnit* cpp_ast, const SemIR::InstNamer* inst_namer,
      llvm::raw_ostream* vlog_stream);

  // Lowers the SemIR::File to LLVM IR. Should only be called once, and handles
  // the main execution loop.
  auto Run() -> std::unique_ptr<llvm::Module>;

  // Create the DICompileUnit metadata for this compilation.
  auto BuildDICompileUnit(llvm::StringRef module_name,
                          llvm::Module& llvm_module,
                          llvm::DIBuilder& di_builder) -> llvm::DICompileUnit*;

  // Creates the Clang `CodeGenerator` to generate LLVM module from imported C++
  // code. Returns null when not importing C++.
  auto CreateCppCodeGenerator() -> std::unique_ptr<clang::CodeGenerator>;

  // Gets a callable's function. Returns nullptr for a builtin.
  auto GetFunction(SemIR::FunctionId function_id) -> llvm::Function* {
    return functions_[function_id.index];
  }

  // Gets a or creates callable's function. Returns nullptr for a builtin.
  auto GetOrCreateFunction(SemIR::FunctionId function_id,
                           SemIR::SpecificId specific_id) -> llvm::Function*;

  // Returns a lowered type for the given type_id.
  auto GetType(SemIR::TypeId type_id) -> llvm::Type* {
    CARBON_CHECK(type_id.has_value(), "Should not be called with `None`");
    CARBON_CHECK(type_id.is_concrete(), "Lowering symbolic type {0}: {1}",
                 type_id, sem_ir().types().GetAsInst(type_id));
    CARBON_CHECK(types_[type_id.index], "Missing type {0}: {1}", type_id,
                 sem_ir().types().GetAsInst(type_id));
    return types_[type_id.index];
  }

  // Returns location information for use with DebugInfo.
  auto GetLocForDI(SemIR::InstId inst_id) -> LocForDI;

  // Returns a lowered value to use for a value of type `type`.
  auto GetTypeAsValue() -> llvm::Constant* {
    return llvm::ConstantStruct::get(GetTypeType());
  }

  // Returns a lowered value to use for a value of int literal type.
  auto GetIntLiteralAsValue() -> llvm::Constant* {
    // TODO: Consider adding a named struct type for integer literals.
    return llvm::ConstantStruct::get(llvm::StructType::get(llvm_context()));
  }

  // Returns a global value for the given instruction.
  auto GetGlobal(SemIR::InstId inst_id, SemIR::SpecificId specific_id)
      -> llvm::Value*;

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
  auto cpp_ast() -> clang::ASTUnit* { return cpp_ast_; }
  auto inst_namer() -> const SemIR::InstNamer* { return inst_namer_; }
  auto global_variables() -> const Map<SemIR::InstId, llvm::GlobalVariable*>& {
    return global_variables_;
  }
  auto printf_int_format_string() -> llvm::Value* {
    return printf_int_format_string_;
  }
  auto SetPrintfIntFormatString(llvm::Value* printf_int_format_string) {
    CARBON_CHECK(!printf_int_format_string_,
                 "PrintInt formatting string already generated");
    printf_int_format_string_ = printf_int_format_string;
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

 private:
  // Builds the declaration for the given function, which should then be cached
  // by the caller.
  auto BuildFunctionDecl(SemIR::FunctionId function_id,
                         SemIR::SpecificId specific_id =
                             SemIR::SpecificId::None) -> llvm::Function*;

  // Builds the definition for the given function. If the function is only a
  // declaration with no definition, does nothing. If this is a generic it'll
  // only be lowered if the specific_id is specified. During this lowering of
  // a generic, more generic functions may be added for lowering.
  auto BuildFunctionDefinition(
      SemIR::FunctionId function_id,
      SemIR::SpecificId specific_id = SemIR::SpecificId::None) -> void;

  // Builds a functions body. Common functionality for all functions.
  auto BuildFunctionBody(
      SemIR::FunctionId function_id, const SemIR::Function& function,
      llvm::Function* llvm_function,
      SemIR::SpecificId specific_id = SemIR::SpecificId::None) -> void;

  // Build the DISubprogram metadata for the given function.
  auto BuildDISubprogram(const SemIR::Function& function,
                         const llvm::Function* llvm_function)
      -> llvm::DISubprogram*;

  // Builds the type for the given instruction, which should then be cached by
  // the caller.
  auto BuildType(SemIR::InstId inst_id) -> llvm::Type*;

  auto BuildVtable(const SemIR::Class& class_info) -> llvm::GlobalVariable*;

  // Records a specific that was lowered for a generic. These are added one
  // by one while lowering their definitions.
  auto AddLoweredSpecificForGeneric(SemIR::GenericId generic_id,
                                    SemIR::SpecificId specific_id) {
    lowered_specifics_[generic_id.index].push_back(specific_id);
  }

  // Initializes and returns a SpecificFunctionFingerprint* instance for a
  // specific. The internal of the fingerprint are populated during and after
  // lowering the function body of that specific.
  auto InitializeFingerprintForSpecific(SemIR::SpecificId specific_id)
      -> SpecificFunctionFingerprint* {
    if (!specific_id.has_value()) {
      return nullptr;
    }
    return &lowered_specific_fingerprint_[specific_id.index];
  }

  // Entry point for coalescing equivalent specifics. Two function definitions,
  // from the same generic, with different specific_ids are considered
  // equivalent if, at the LLVM level, one can be replaced with the other, with
  // no change in behavior. All LLVM types and instructions must be equivalent.
  auto CoalesceEquivalentSpecifics() -> void;

  // While coalescing specifics, returns whether the function types for two
  // specifics are equivalent. This uses a fingerprint generated for each
  // function type.
  auto AreFunctionTypesEquivalent(SemIR::SpecificId specific_id1,
                                  SemIR::SpecificId specific_id2) -> bool;

  // While coalescing specifics, compare the function bodies for two specifics.
  // This uses fingerprints generated during lowering of the function body.
  // The `visited_equivalent_specifics` parameter is used to track cycles in
  // the function callgraph, and will also return equivalent pairs of specifics
  // found, if the two specifics given as arguments are found to be equivalent.
  auto AreFunctionBodiesEquivalent(
      SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
      Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>&
          visited_equivalent_specifics) -> bool;

  // Given an equivalent pair of specifics, updates the canonical specific to
  // use for each of the two Specifics found to be equivalent, replaces all
  // uses of one specific with the canonical one, and adds the non-canonical
  // specific to specifics_to_delete.
  auto ProcessSpecificEquivalence(
      std::pair<SemIR::SpecificId, SemIR::SpecificId> pair,
      llvm::SmallVector<SemIR::SpecificId>& specifics_to_delete) -> void;

  // Checks if two specific_ids are equivalent and also reduces the equivalence
  // chains/paths. This update ensures the canonical specific is always "one
  // hop away".
  auto IsKnownEquivalence(SemIR::SpecificId specific_id1,
                          SemIR::SpecificId specific_id2) -> bool;

  // Inserts a pair into a set of pairs in canonical form. Also implicitly
  // checks entry already existed if it cannot be inserted.
  auto InsertPair(
      SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
      Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
      -> bool;

  // Checks if a pair is contained into a set of pairs, in canonical form.
  auto ContainsPair(
      SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
      const Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
      -> bool;

  // State for building the LLVM IR.
  llvm::LLVMContext* llvm_context_;
  std::unique_ptr<llvm::Module> llvm_module_;

  // The filesystem for source code.
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs_;

  // State for building the LLVM IR debug info metadata.
  llvm::DIBuilder di_builder_;

  // The DICompileUnit, if any - null implies debug info is not being emitted.
  llvm::DICompileUnit* di_compile_unit_;

  // The trees are only provided when debug info should be emitted.
  std::optional<llvm::ArrayRef<Parse::GetTreeAndSubtreesFn>>
      tree_and_subtrees_getters_for_debug_info_;

  // The input SemIR.
  const SemIR::File* const sem_ir_;

  // A mutable Clang AST is necessary for lowering since using the AST in lower
  // modifies it.
  clang::ASTUnit* cpp_ast_;

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
  // Vector indexes correspond to `FunctionId` indexes. We resize this directly
  // to the correct size.
  llvm::SmallVector<llvm::Function*, 0> functions_;

  // Maps specific callables to lowered functions. Vector indexes correspond to
  // `SpecificId` indexes. We resize this directly to the correct size.
  llvm::SmallVector<llvm::Function*, 0> specific_functions_;

  // Maps which specific functions are generics that need to have their
  // definitions lowered after the lowering of other definitions.
  // This list may grow while lowering generic definitions from this list.
  // The list uses the `SpecificId` to index into specific_functions_.
  llvm::SmallVector<std::pair<SemIR::FunctionId, SemIR::SpecificId>, 10>
      specific_function_definitions_;

  // Provides lowered versions of types.
  // Vector indexes correspond to `TypeId` indexes for non-symbolic types. We
  // resize this directly to the (often large) correct size.
  llvm::SmallVector<llvm::Type*, 0> types_;

  // Lowered version of the builtin type `type`.
  llvm::StructType* type_type_ = nullptr;

  // Maps constants to their lowered values.
  // Vector indexes correspond to `InstId` indexes for constant instructions. We
  // resize this directly to the (often large) correct size.
  llvm::SmallVector<llvm::Constant*, 0> constants_;

  // Maps global variables to their lowered variant.
  Map<SemIR::InstId, llvm::GlobalVariable*> global_variables_;

  // Global format string for `printf.int.format` used by the PrintInt builtin.
  llvm::Value* printf_int_format_string_ = nullptr;

  // For a generic function, keep track of the specifics for which LLVM
  // function declarations were created. Those can be retrieved then from
  // `specific_functions_`. We resize this to the correct size. Vector indexes
  // correspond to `GenericId` indexes.
  llvm::SmallVector<llvm::SmallVector<SemIR::SpecificId>, 0> lowered_specifics_;

  // For specifics that exist in lowered_specifics, a hash of their function
  // type information: return and parameter types. We resize this to the
  // correct size. Vector indexes correspond to `SpecificId` indexes.
  // TODO: Hashing all members of `FunctionTypeInfo` may not be necessary.
  llvm::SmallVector<llvm::BLAKE3Result<32>, 0>
      lowered_specifics_type_fingerprint_;

  // This is initialized and populated while lowering a specific.
  // We resize this to the correct size. Vector indexes correspond to
  // `SpecificId` indexes.
  llvm::SmallVector<SpecificFunctionFingerprint, 0>
      lowered_specific_fingerprint_;

  // Equivalent specifics that have been found. For each specific, this points
  // to the canonical equivalent specific, which may also be self. We currently
  // define the canonical specific as the one with the lowest
  // `SpecificId.index`.
  //
  // We resize this to the correct size and initialize to `SpecificId::None`,
  // which defines that there is no other equivalent specific to this
  // `SpecificId`. Vector indexes correspond to `SpecificId` indexes.
  llvm::SmallVector<SemIR::SpecificId, 0> equivalent_specifics_;

  // Non-equivalent specifics found.
  // TODO: Revisit this due to its quadratic space growth.
  Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>
      non_equivalent_specifics_;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
