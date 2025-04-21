// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
#define CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_

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

  struct SpecificFunctionFingerprint {
    llvm::BLAKE3Result<32UL> function_common_fingerprint;
    llvm::BLAKE3Result<32UL> function_specific_fingerprint;
    llvm::SmallVector<SemIR::SpecificId> calls;
  };

  explicit FileContext(
      llvm::LLVMContext& llvm_context,
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

 private:
  struct FunctionTypeInfo {
    llvm::FunctionType* type;
    llvm::SmallVector<SemIR::InstId> param_inst_ids;
    llvm::Type* return_type = nullptr;
    SemIR::InstId return_param_id = SemIR::InstId::None;

    // Compares the types of two functions. This is intended to be used for
    // specifics of the same generic.
    auto Equals(FunctionTypeInfo& other) -> bool {
      // Two specifics of the same generic may not have the same number of
      // params. It's possible one specific uses a return slot (so has one
      // more param) while another does not.
      if (param_inst_ids.size() != other.param_inst_ids.size() ||
          return_param_id != other.return_param_id || type != other.type ||
          return_type != other.return_type) {
        return false;
      }

      for (const auto [param_id, other_param_id] :
           llvm::zip_equal(param_inst_ids, other.param_inst_ids)) {
        if (param_id != other_param_id) {
          return false;
        }
      }
      return true;
    }
  };

  // Retrieve various features of the function's type useful for constructing
  // the `llvm::Type` for the `llvm::Function`. If any part of the type can't be
  // manifest (eg: incomplete return or parameter types), then the result is as
  // if the type was `void()`.
  auto BuildFunctionTypeInfo(const SemIR::Function& function,
                             SemIR::SpecificId specific_id) -> FunctionTypeInfo;

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

  // Builds the global for the given instruction, which should then be cached by
  // the caller.
  auto BuildGlobalVariableDecl(SemIR::VarStorage var_storage)
      -> llvm::GlobalVariable*;

  auto BuildVtable(const SemIR::Class& class_info) -> llvm::GlobalVariable*;

  auto AddLoweredSpecificForGeneric(SemIR::GenericId generic_id,
                                    SemIR::SpecificId specific_id) {
    lowered_specifics_[generic_id.index].push_back(specific_id);
  }

  auto InitializeFingerprintForSpecific(SemIR::SpecificId specific_id)
      -> SpecificFunctionFingerprint* {
    if (!specific_id.has_value()) {
      return nullptr;
    }
    return &lowered_specific_fingerprint_[specific_id.index];
  }

  auto CoalesceEquivalentSpecifics() -> void;
  auto CheckTypeEquivalence(SemIR::SpecificId specific1,
                            SemIR::SpecificId specific2) -> bool;
  auto CheckBodyEquivalence(
      SemIR::SpecificId specific1, SemIR::SpecificId specific2,
      Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>&
          visited_equivalent_specifics,
      Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>&
          visited_equivalent_specifics_flipped) -> bool;
  auto AddOrUpdateSpecificEquivalence(
      std::pair<SemIR::SpecificId, SemIR::SpecificId>& pair,
      Set<SemIR::SpecificId>& specifics_to_delete) -> void;
  auto CheckExistingEquivalence(SemIR::SpecificId, SemIR::SpecificId) -> bool;
  auto DeleteFunctionSpecific(SemIR::SpecificId to_replace,
                              SemIR::SpecificId replace_with) -> void;

  // State for building the LLVM IR.
  llvm::LLVMContext* llvm_context_;
  std::unique_ptr<llvm::Module> llvm_module_;

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
  // function declarations were created. Those can be retrieved then via
  // from specific_functions_, via specific_functions_[specific_id.index].
  // We resize this to the correct size. Indexed by generic_id.index.
  llvm::SmallVector<llvm::SmallVector<SemIR::SpecificId>, 0> lowered_specifics_;

  // For specifics that exist in lowered_specifics, a hash of their function
  // type information: return and parameter types. We resize this to the
  // correct size. Indexed by specific_id.index.
  // TODO: Hashing all members of FunctionTypeInfo may not be necessary.
  llvm::SmallVector<llvm::BLAKE3Result<32UL>, 0>
      lowered_specifics_type_fingerprint_;

  // This is initialized and populated while lowering a specific.
  // We resize this to the correct size. Indexed by specific_id.index.
  llvm::SmallVector<SpecificFunctionFingerprint, 0>
      lowered_specific_fingerprint_;

  // Equivalent specifics found: for each specific point to the
  // canonical equivalent specific
  llvm::SmallVector<SemIR::SpecificId> equivalent_specifics_;
  // Non-equivalent specifics found.
  Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>
      non_equivalent_specifics_;
  // Track whether a specific was replaced by another. We resize it to the
  // correct size and initialize with all false. Indexed by specific_id.index.
  llvm::SmallVector<bool, 0> is_replaced_specific_;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_FILE_CONTEXT_H_
