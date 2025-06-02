// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/file_context.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "clang/CodeGen/ModuleBuilder.h"
#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/constant.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/lower/mangler.h"
#include "toolchain/sem_ir/absolute_node_id.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

FileContext::FileContext(Context& context, const SemIR::File& sem_ir,
                         const SemIR::InstNamer* inst_namer,
                         llvm::raw_ostream* vlog_stream)
    : context_(&context),
      sem_ir_(&sem_ir),
      inst_namer_(inst_namer),
      vlog_stream_(vlog_stream) {
  // Initialization that relies on invariants of the class.
  cpp_code_generator_ = CreateCppCodeGenerator();
  CARBON_CHECK(!sem_ir.has_errors(),
               "Generating LLVM IR from invalid SemIR::File is unsupported.");
}

// TODO: Move this to lower.cpp.
auto FileContext::PrepareToLower() -> void {
  if (cpp_code_generator_) {
    // Clang code generation should not actually modify the AST, but isn't
    // const-correct.
    cpp_code_generator_->Initialize(
        const_cast<clang::ASTContext&>(cpp_ast()->getASTContext()));
  }

  // Lower all types that were required to be complete.
  types_.resize(sem_ir_->insts().size());
  for (auto type_id : sem_ir_->types().complete_types()) {
    if (type_id.index >= 0) {
      types_[type_id.index] = BuildType(sem_ir_->types().GetInstId(type_id));
    }
  }

  // Lower function declarations.
  functions_.resize_for_overwrite(sem_ir_->functions().size());
  for (auto [id, _] : sem_ir_->functions().enumerate()) {
    functions_[id.index] = BuildFunctionDecl(id);
  }

  // Specific functions are lowered when we emit a reference to them.
  specific_functions_.resize(sem_ir_->specifics().size());
  // Additional data stored for specifics, for when attempting to coalesce.
  // Indexed by `GenericId`.
  lowered_specifics_.resize(sem_ir_->generics().size());
  // Indexed by `SpecificId`.
  lowered_specifics_type_fingerprint_.resize(sem_ir_->specifics().size());
  lowered_specific_fingerprint_.resize(sem_ir_->specifics().size());
  equivalent_specifics_.resize(sem_ir_->specifics().size(),
                               SemIR::SpecificId::None);

  // Lower constants.
  constants_.resize(sem_ir_->insts().size());
  LowerConstants(*this, constants_);
}

// TODO: Move this to lower.cpp.
auto FileContext::LowerDefinitions() -> void {
  for (const auto& class_info : sem_ir_->classes().values()) {
    if (auto* llvm_vtable = BuildVtable(class_info)) {
      global_variables_.Insert(class_info.vtable_id, llvm_vtable);
    }
  }

  // Lower global variable definitions.
  // TODO: Storing both a `constants_` array and a separate `global_variables_`
  // map is redundant.
  for (auto inst_id :
       sem_ir().inst_blocks().Get(sem_ir().top_inst_block_id())) {
    // Only `VarStorage` indicates a global variable declaration in the
    // top instruction block.
    if (auto var = sem_ir().insts().TryGetAs<SemIR::VarStorage>(inst_id)) {
      // Get the global variable declaration. We created this when lowering the
      // constant unless the variable is unnamed, in which case we need to
      // create it now.
      llvm::GlobalVariable* llvm_var = nullptr;
      if (sem_ir().constant_values().Get(inst_id).is_constant()) {
        llvm_var = cast<llvm::GlobalVariable>(
            GetGlobal(inst_id, SemIR::SpecificId::None));
      } else {
        llvm_var = BuildGlobalVariableDecl(*var);
      }

      // Convert the declaration of this variable into a definition by adding an
      // initializer.
      global_variables_.Insert(inst_id, llvm_var);
      llvm_var->setInitializer(
          llvm::Constant::getNullValue(llvm_var->getValueType()));
    }
  }

  // Lower function definitions.
  for (auto [id, _] : sem_ir_->functions().enumerate()) {
    BuildFunctionDefinition(id);
  }

  // Lower function definitions for generics.
  // This cannot be a range-based loop, as new definitions can be added
  // while building other definitions.
  // NOLINTNEXTLINE
  for (size_t i = 0; i != specific_function_definitions_.size(); ++i) {
    auto [function_id, specific_id] = specific_function_definitions_[i];
    BuildFunctionDefinition(function_id, specific_id);
  }

  // Append `__global_init` to `llvm::global_ctors` to initialize global
  // variables.
  if (sem_ir().global_ctor_id().has_value()) {
    llvm::appendToGlobalCtors(llvm_module(),
                              GetFunction(sem_ir().global_ctor_id()),
                              /*Priority=*/0);
  }

  if (cpp_code_generator_) {
    // Clang code generation should not actually modify the AST, but isn't
    // const-correct.
    cpp_code_generator_->HandleTranslationUnit(
        const_cast<clang::ASTContext&>(cpp_ast()->getASTContext()));
    bool link_error = llvm::Linker::linkModules(
        /*Dest=*/llvm_module(),
        /*Src=*/std::unique_ptr<llvm::Module>(
            cpp_code_generator_->ReleaseModule()));
    CARBON_CHECK(!link_error);
  }
}

auto FileContext::Finalize() -> void {
  // Find equivalent specifics (from the same generic), replace all uses and
  // remove duplicately lowered function definitions.
  CoalesceEquivalentSpecifics();
}

auto FileContext::InsertPair(
    SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
    Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
    -> bool {
  if (specific_id1.index > specific_id2.index) {
    std::swap(specific_id1.index, specific_id2.index);
  }
  auto insert_result =
      set_of_pairs.Insert(std::make_pair(specific_id1, specific_id2));
  return insert_result.is_inserted();
}

auto FileContext::ContainsPair(
    SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
    const Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
    -> bool {
  if (specific_id1.index > specific_id2.index) {
    std::swap(specific_id1.index, specific_id2.index);
  }
  return set_of_pairs.Contains(std::make_pair(specific_id1, specific_id2));
}

auto FileContext::CoalesceEquivalentSpecifics() -> void {
  for (auto& specifics : lowered_specifics_) {
    // i cannot be unsigned due to the comparison with a negative number when
    // the specifics vector is empty.
    for (int i = 0; i < static_cast<int>(specifics.size()) - 1; ++i) {
      // This specific was already replaced, skip it.
      if (equivalent_specifics_[specifics[i].index].has_value() &&
          equivalent_specifics_[specifics[i].index] != specifics[i]) {
        specifics[i] = specifics[specifics.size() - 1];
        specifics.pop_back();
        --i;
        continue;
      }
      // TODO: Improve quadratic behavior by using a single hash based on
      // `lowered_specifics_type_fingerprint_` and `common_fingerprint`.
      for (int j = i + 1; j < static_cast<int>(specifics.size()); ++j) {
        // When the specific was already replaced, skip it.
        if (equivalent_specifics_[specifics[j].index].has_value() &&
            equivalent_specifics_[specifics[j].index] != specifics[j]) {
          specifics[j] = specifics[specifics.size() - 1];
          specifics.pop_back();
          --j;
          continue;
        }

        // When the two specifics are not equivalent due to the function type
        // info stored in lowered_specifics_types, mark non-equivalance. This
        // can be reused to short-cut another path and continue the search for
        // other equivalences.
        if (!AreFunctionTypesEquivalent(specifics[i], specifics[j])) {
          InsertPair(specifics[i], specifics[j], non_equivalent_specifics_);
          continue;
        }

        Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>
            visited_equivalent_specifics;
        InsertPair(specifics[i], specifics[j], visited_equivalent_specifics);
        // Function type information matches; check usages inside the function
        // body that are dependent on the specific. This information has been
        // stored in lowered_states while lowering each function body.
        if (AreFunctionBodiesEquivalent(specifics[i], specifics[j],
                                        visited_equivalent_specifics)) {
          // When processing equivalences, we may change the canonical specific
          // multiple times, so we don't delete replaced specifics until the
          // end.
          llvm::SmallVector<SemIR::SpecificId> specifics_to_delete;
          visited_equivalent_specifics.ForEach(
              [&](std::pair<SemIR::SpecificId, SemIR::SpecificId>
                      equivalent_entry) {
                CARBON_VLOG("Found equivalent specifics: {0}, {1}",
                            equivalent_entry.first, equivalent_entry.second);
                ProcessSpecificEquivalence(equivalent_entry,
                                           specifics_to_delete);
              });

          // Delete function bodies for already replaced functions.
          for (auto specific_id : specifics_to_delete) {
            specific_functions_[specific_id.index]->eraseFromParent();
            specific_functions_[specific_id.index] =
                specific_functions_[equivalent_specifics_[specific_id.index]
                                        .index];
          }

          // Removed the replaced specific from the list of emitted specifics.
          // Only the top level, since the others are somewhere else in the
          // vector, they will be found and removed during processing.
          specifics[j] = specifics[specifics.size() - 1];
          specifics.pop_back();
          --j;
        } else {
          // Only mark non-equivalence based on state for starting specifics.
          InsertPair(specifics[i], specifics[j], non_equivalent_specifics_);
        }
      }
    }
  }
}

auto FileContext::ProcessSpecificEquivalence(
    std::pair<SemIR::SpecificId, SemIR::SpecificId> pair,
    llvm::SmallVector<SemIR::SpecificId>& specifics_to_delete) -> void {
  auto [specific_id1, specific_id2] = pair;
  CARBON_CHECK(specific_id1.has_value() && specific_id2.has_value(),
               "Expected values in equivalence check");

  auto get_canon = [&](SemIR::SpecificId specific_id) {
    return equivalent_specifics_[specific_id.index].has_value()
               ? std::make_pair(
                     equivalent_specifics_[specific_id.index],
                     (equivalent_specifics_[specific_id.index] != specific_id))
               : std::make_pair(specific_id, false);
  };
  auto [canon_id1, replaced_before1] = get_canon(specific_id1);
  auto [canon_id2, replaced_before2] = get_canon(specific_id2);

  if (canon_id1 == canon_id2) {
    // Already equivalent, there was a previous replacement.
    return;
  }

  if (canon_id1.index >= canon_id2.index) {
    // Prefer the earlier index for canonical values.
    std::swap(canon_id1, canon_id2);
    std::swap(replaced_before1, replaced_before2);
  }

  // Update equivalent_specifics_ for all. This is used as an indicator that
  // this specific_id may be the canonical one when reducing the equivalence
  // chains in `IsKnownEquivalence`.
  equivalent_specifics_[specific_id1.index] = canon_id1;
  equivalent_specifics_[specific_id2.index] = canon_id1;
  specific_functions_[canon_id2.index]->replaceAllUsesWith(
      specific_functions_[canon_id1.index]);
  if (!replaced_before2) {
    specifics_to_delete.push_back(canon_id2);
  }
}

auto FileContext::IsKnownEquivalence(SemIR::SpecificId specific_id1,
                                     SemIR::SpecificId specific_id2) -> bool {
  if (!equivalent_specifics_[specific_id1.index].has_value() ||
      !equivalent_specifics_[specific_id2.index].has_value()) {
    return false;
  }

  auto update_equivalent_specific = [&](SemIR::SpecificId specific_id) {
    llvm::SmallVector<SemIR::SpecificId> stack;
    SemIR::SpecificId specific_to_update = specific_id;
    while (equivalent_specifics_[equivalent_specifics_[specific_to_update.index]
                                     .index] !=
           equivalent_specifics_[specific_to_update.index]) {
      stack.push_back(specific_to_update);
      specific_to_update = equivalent_specifics_[specific_to_update.index];
    }
    for (auto specific : llvm::reverse(stack)) {
      equivalent_specifics_[specific.index] =
          equivalent_specifics_[equivalent_specifics_[specific.index].index];
    }
  };

  update_equivalent_specific(specific_id1);
  update_equivalent_specific(specific_id2);

  return equivalent_specifics_[specific_id1.index] ==
         equivalent_specifics_[specific_id2.index];
}

auto FileContext::AreFunctionTypesEquivalent(SemIR::SpecificId specific_id1,
                                             SemIR::SpecificId specific_id2)
    -> bool {
  CARBON_CHECK(specific_id1.has_value() && specific_id2.has_value());
  return lowered_specifics_type_fingerprint_[specific_id1.index] ==
         lowered_specifics_type_fingerprint_[specific_id2.index];
}

auto FileContext::AreFunctionBodiesEquivalent(
    SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
    Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>&
        visited_equivalent_specifics) -> bool {
  llvm::SmallVector<std::pair<SemIR::SpecificId, SemIR::SpecificId>> worklist;
  worklist.push_back({specific_id1, specific_id2});

  while (!worklist.empty()) {
    auto outer_pair = worklist.pop_back_val();
    auto [specific_id1, specific_id2] = outer_pair;

    auto state1 = lowered_specific_fingerprint_[specific_id1.index];
    auto state2 = lowered_specific_fingerprint_[specific_id2.index];
    if (state1.common_fingerprint != state2.common_fingerprint) {
      InsertPair(specific_id1, specific_id2, non_equivalent_specifics_);
      return false;
    }
    if (state1.specific_fingerprint == state2.specific_fingerprint) {
      continue;
    }

    // A size difference should have been detected by the common fingerprint.
    CARBON_CHECK(state1.calls.size() == state2.calls.size(),
                 "Number of specific calls expected to be the same.");

    for (auto [state1_call, state2_call] :
         llvm::zip(state1.calls, state2.calls)) {
      if (state1_call != state2_call) {
        if (ContainsPair(state1_call, state2_call, non_equivalent_specifics_)) {
          return false;
        }
        if (IsKnownEquivalence(state1_call, state2_call)) {
          continue;
        }
        if (!InsertPair(state1_call, state2_call,
                        visited_equivalent_specifics)) {
          continue;
        }
        // Leave the added equivalence pair in place and continue.
        worklist.push_back({state1_call, state2_call});
      }
    }
  }
  return true;
}

auto FileContext::CreateCppCodeGenerator()
    -> std::unique_ptr<clang::CodeGenerator> {
  if (!cpp_ast()) {
    return nullptr;
  }

  RawStringOstream clang_module_name_stream;
  clang_module_name_stream << llvm_module().getName() << ".clang";

  // Do not emit Clang's name and version as the creator of the output file.
  cpp_code_gen_options_.EmitVersionIdentMetadata = false;

  return std::unique_ptr<clang::CodeGenerator>(clang::CreateLLVMCodeGen(
      cpp_ast()->getASTContext().getDiagnostics(),
      clang_module_name_stream.TakeStr(), context().file_system(),
      cpp_header_search_options_, cpp_preprocessor_options_,
      cpp_code_gen_options_, llvm_context()));
}

auto FileContext::GetGlobal(SemIR::InstId inst_id,
                            SemIR::SpecificId specific_id) -> llvm::Value* {
  auto const_id = GetConstantValueInSpecific(sem_ir(), specific_id, inst_id);
  CARBON_CHECK(const_id.is_concrete(), "Missing value: {0} {1} {2}", inst_id,
               specific_id, sem_ir().insts().Get(inst_id));
  auto const_inst_id = sem_ir().constant_values().GetInstId(const_id);
  auto* const_value = constants_[const_inst_id.index];

  // For value expressions and initializing expressions, the value produced by
  // a constant instruction is a value representation of the constant. For
  // initializing expressions, `FinishInit` will perform a copy if needed.
  switch (auto cat = SemIR::GetExprCategory(sem_ir(), const_inst_id)) {
    case SemIR::ExprCategory::Value:
    case SemIR::ExprCategory::Initializing:
      break;

    case SemIR::ExprCategory::DurableRef:
    case SemIR::ExprCategory::EphemeralRef:
      // Constant reference expressions lower to an address.
      return const_value;

    case SemIR::ExprCategory::NotExpr:
    case SemIR::ExprCategory::Error:
    case SemIR::ExprCategory::Mixed:
      CARBON_FATAL("Unexpected category {0} for lowered constant {1}", cat,
                   sem_ir().insts().Get(const_inst_id));
  };

  auto value_rep = SemIR::ValueRepr::ForType(
      sem_ir(), sem_ir().insts().Get(const_inst_id).type_id());
  if (value_rep.kind != SemIR::ValueRepr::Pointer) {
    return const_value;
  }

  // The value representation is a pointer. Generate a variable to hold the
  // value, or find and reuse an existing one.
  if (auto result = global_variables().Lookup(const_inst_id)) {
    return result.value();
  }

  // Include both the name of the constant, if any, and the point of use in
  // the name of the variable.
  llvm::StringRef const_name;
  llvm::StringRef use_name;
  if (inst_namer_) {
    const_name = inst_namer_->GetUnscopedNameFor(const_inst_id);
    use_name = inst_namer_->GetUnscopedNameFor(inst_id);
  }

  // We always need to give the global a name even if the instruction namer
  // doesn't have one to use.
  if (const_name.empty()) {
    const_name = "const";
  }
  if (use_name.empty()) {
    use_name = "anon";
  }
  llvm::StringRef sep = (use_name[0] == '.') ? "" : ".";

  auto* global_variable = new llvm::GlobalVariable(
      llvm_module(), GetType(sem_ir().GetPointeeType(value_rep.type_id)),
      /*isConstant=*/true, llvm::GlobalVariable::InternalLinkage, const_value,
      const_name + sep + use_name);

  global_variables_.Insert(const_inst_id, global_variable);
  return global_variable;
}

auto FileContext::GetOrCreateFunction(SemIR::FunctionId function_id,
                                      SemIR::SpecificId specific_id)
    -> llvm::Function* {
  // Non-generic functions are declared eagerly.
  if (!specific_id.has_value()) {
    return GetFunction(function_id);
  }

  if (auto* result = specific_functions_[specific_id.index]) {
    return result;
  }

  auto* result = BuildFunctionDecl(function_id, specific_id);
  // TODO: Add this function to a list of specific functions whose definitions
  // we need to emit.
  specific_functions_[specific_id.index] = result;
  // TODO: Use this to generate definitions for these functions.
  specific_function_definitions_.push_back({function_id, specific_id});
  return result;
}

auto FileContext::BuildFunctionTypeInfo(const SemIR::Function& function,
                                        SemIR::SpecificId specific_id)
    -> FunctionTypeInfo {
  const auto return_info =
      SemIR::ReturnTypeInfo::ForFunction(sem_ir(), function, specific_id);

  if (!return_info.is_valid()) {
    // The return type has not been completed, create a trivial type instead.
    return {.type =
                llvm::FunctionType::get(llvm::Type::getVoidTy(llvm_context()),
                                        /*isVarArg=*/false)};
  }

  auto get_llvm_type = [&](SemIR::TypeId type_id) -> llvm::Type* {
    if (!type_id.has_value()) {
      return nullptr;
    }
    return GetType(type_id);
  };

  // TODO: expose the `Call` parameter patterns in `Function`, and use them here
  // instead of reconstructing them via the syntactic parameter lists.
  auto implicit_param_patterns =
      sem_ir().inst_blocks().GetOrEmpty(function.implicit_param_patterns_id);
  auto param_patterns =
      sem_ir().inst_blocks().GetOrEmpty(function.param_patterns_id);

  auto* return_type = get_llvm_type(return_info.type_id);

  llvm::SmallVector<llvm::Type*> param_types;
  // Compute the return type to use for the LLVM function. If the initializing
  // representation doesn't produce a value, set the return type to void.
  // TODO: For the `Run` entry point, remap return type to i32 if it doesn't
  // return a value.
  llvm::Type* function_return_type =
      (return_info.is_valid() &&
       return_info.init_repr.kind == SemIR::InitRepr::ByCopy)
          ? return_type
          : llvm::Type::getVoidTy(llvm_context());

  // TODO: Consider either storing `param_inst_ids` somewhere so that we can
  // reuse it from `BuildFunctionDefinition` and when building calls, or factor
  // out a mechanism to compute the mapping between parameters and arguments on
  // demand.
  llvm::SmallVector<SemIR::InstId> param_inst_ids;
  auto max_llvm_params = (return_info.has_return_slot() ? 1 : 0) +
                         implicit_param_patterns.size() + param_patterns.size();
  param_types.reserve(max_llvm_params);
  param_inst_ids.reserve(max_llvm_params);
  auto return_param_id = SemIR::InstId::None;
  if (return_info.has_return_slot()) {
    param_types.push_back(
        llvm::PointerType::get(llvm_context(), /*AddressSpace=*/0));
    return_param_id = function.return_slot_pattern_id;
    param_inst_ids.push_back(return_param_id);
  }
  for (auto param_pattern_id : llvm::concat<const SemIR::InstId>(
           implicit_param_patterns, param_patterns)) {
    auto param_pattern_info = SemIR::Function::GetParamPatternInfoFromPatternId(
        sem_ir(), param_pattern_id);
    if (!param_pattern_info) {
      continue;
    }
    auto param_type_id = ExtractScrutineeType(
        sem_ir(), SemIR::GetTypeOfInstInSpecific(sem_ir(), specific_id,
                                                 param_pattern_info->inst_id));
    CARBON_CHECK(
        !param_type_id.AsConstantId().is_symbolic(),
        "Found symbolic type id after resolution when lowering type {0}.",
        param_pattern_info->inst.type_id);
    switch (auto value_rep = SemIR::ValueRepr::ForType(sem_ir(), param_type_id);
            value_rep.kind) {
      case SemIR::ValueRepr::Unknown:
        // This parameter type is incomplete. Fallback to describing the
        // function type as `void()`.
        return {.type = llvm::FunctionType::get(
                    llvm::Type::getVoidTy(llvm_context()),
                    /*isVarArg=*/false)};
      case SemIR::ValueRepr::None:
        break;
      case SemIR::ValueRepr::Copy:
      case SemIR::ValueRepr::Custom:
      case SemIR::ValueRepr::Pointer:
        auto* param_types_to_add = get_llvm_type(value_rep.type_id);
        param_types.push_back(param_types_to_add);
        param_inst_ids.push_back(param_pattern_id);
        break;
    }
  }
  return {.type = llvm::FunctionType::get(function_return_type, param_types,
                                          /*isVarArg=*/false),
          .param_inst_ids = std::move(param_inst_ids),
          .return_type = return_type,
          .return_param_id = return_param_id};
}

auto FileContext::BuildFunctionDecl(SemIR::FunctionId function_id,
                                    SemIR::SpecificId specific_id)
    -> llvm::Function* {
  const auto& function = sem_ir().functions().Get(function_id);

  // Don't lower generic functions. Note that associated functions in interfaces
  // have `Self` in scope, so are implicitly generic functions.
  if (function.generic_id.has_value() && !specific_id.has_value()) {
    return nullptr;
  }

  // Don't lower builtins.
  if (function.builtin_function_kind != SemIR::BuiltinFunctionKind::None) {
    return nullptr;
  }

  // TODO: Consider tracking whether the function has been used, and only
  // lowering it if it's needed.

  auto function_type_info = BuildFunctionTypeInfo(function, specific_id);

  auto linkage = specific_id.has_value() ? llvm::Function::LinkOnceODRLinkage
                                         : llvm::Function::ExternalLinkage;

  Mangler m(*this);
  std::string mangled_name = m.Mangle(function_id, specific_id);

  // Create a unique fingerprint for the function type.
  // For now, compute the function type fingerprint only for specifics, though
  // we might need it for all functions in order to create a canonical
  // fingerprint across translation units.
  if (specific_id.has_value()) {
    llvm::BLAKE3 function_type_fingerprint;
    RawStringOstream os;
    function_type_info.type->print(os);
    function_type_fingerprint.update(os.TakeStr());
    function_type_fingerprint.final(
        lowered_specifics_type_fingerprint_[specific_id.index]);
  }

  auto* llvm_function = llvm::Function::Create(function_type_info.type, linkage,
                                               mangled_name, llvm_module());

  CARBON_CHECK(llvm_function->getName() == mangled_name,
               "Mangled name collision: {0}", mangled_name);

  // Set up parameters and the return slot.
  for (auto [inst_id, arg] : llvm::zip_equal(function_type_info.param_inst_ids,
                                             llvm_function->args())) {
    auto name_id = SemIR::NameId::None;
    if (inst_id == function_type_info.return_param_id) {
      name_id = SemIR::NameId::ReturnSlot;
      arg.addAttr(llvm::Attribute::getWithStructRetType(
          llvm_context(), function_type_info.return_type));
    } else {
      name_id = SemIR::GetPrettyNameFromPatternId(sem_ir(), inst_id);
    }
    arg.setName(sem_ir().names().GetIRBaseName(name_id));
  }

  return llvm_function;
}

auto FileContext::BuildFunctionDefinition(SemIR::FunctionId function_id,
                                          SemIR::SpecificId specific_id)
    -> void {
  const auto& function = sem_ir().functions().Get(function_id);
  const auto& body_block_ids = function.body_block_ids;
  if (body_block_ids.empty() &&
      (!function.cpp_decl || !function.cpp_decl->isDefined())) {
    // Function is probably defined in another file; not an error.
    return;
  }

  llvm::Function* llvm_function;
  if (specific_id.has_value()) {
    llvm_function = specific_functions_[specific_id.index];
  } else {
    llvm_function = GetFunction(function_id);
    if (!llvm_function) {
      // We chose not to lower this function at all, for example because it's a
      // generic function.
      return;
    }
  }

  // For non-generics we do not lower. For generics, the llvm function was
  // created via GetOrCreateFunction prior to this when building the
  // declaration.
  BuildFunctionBody(function_id, function, llvm_function, specific_id);
}

auto FileContext::BuildFunctionBody(SemIR::FunctionId function_id,
                                    const SemIR::Function& function,
                                    llvm::Function* llvm_function,
                                    SemIR::SpecificId specific_id) -> void {
  const auto& body_block_ids = function.body_block_ids;
  CARBON_DCHECK(llvm_function, "LLVM Function not found when lowering body.");

  if (function.cpp_decl) {
    // TODO: To support recursive inline functions, collect all calls to
    // `HandleTopLevelDecl()` in a custom `ASTConsumer` configured in the
    // `ASTUnit`, and replay them in lowering in the `CodeGenerator`. See
    // https://discord.com/channels/655572317891461132/768530752592805919/1370509111585935443
    clang::FunctionDecl* cpp_def = function.cpp_decl->getDefinition();
    CARBON_DCHECK(cpp_def, "No Clang function body found during lowering");

    // Create the LLVM function (`CodeGenModule::GetOrCreateLLVMFunction()`) so
    // that code generation (`CodeGenModule::EmitGlobal()`) would see this
    // function name (`CodeGenModule::getMangledName()`), and will generate its
    // definition.
    llvm::Constant* function_address =
        cpp_code_generator_->GetAddrOfGlobal(clang::GlobalDecl(cpp_def),
                                             /*isForDefinition=*/false);
    CARBON_DCHECK(function_address);

    // Emit the function code.
    cpp_code_generator_->HandleTopLevelDecl(clang::DeclGroupRef(cpp_def));
    return;
  }

  CARBON_DCHECK(!body_block_ids.empty(),
                "No function body blocks found during lowering.");

  // Store which specifics were already lowered (with definitions) for each
  // generic.
  if (function.generic_id.has_value() && specific_id.has_value()) {
    AddLoweredSpecificForGeneric(function.generic_id, specific_id);
  }

  FunctionContext function_lowering(
      *this, llvm_function, specific_id,
      InitializeFingerprintForSpecific(specific_id),
      BuildDISubprogram(function, llvm_function), vlog_stream_);

  // Add parameters to locals.
  // TODO: This duplicates the mapping between sem_ir instructions and LLVM
  // function parameters that was already computed in BuildFunctionDecl.
  // We should only do that once.
  auto call_param_ids =
      sem_ir().inst_blocks().GetOrEmpty(function.call_params_id);
  int param_index = 0;

  // TODO: Find a way to ensure this code and the function-call lowering use
  // the same parameter ordering.

  // Lowers the given parameter. Must be called in LLVM calling convention
  // parameter order.
  auto lower_param = [&](SemIR::InstId param_id) {
    // Get the value of the parameter from the function argument.
    auto param_inst = sem_ir().insts().GetAs<SemIR::AnyParam>(param_id);
    llvm::Value* param_value;

    if (SemIR::ValueRepr::ForType(sem_ir(), param_inst.type_id).kind !=
        SemIR::ValueRepr::None) {
      param_value = llvm_function->getArg(param_index);
      ++param_index;
    } else {
      param_value = llvm::PoisonValue::get(GetType(
          SemIR::GetTypeOfInstInSpecific(sem_ir(), specific_id, param_id)));
    }
    // The value of the parameter is the value of the argument.
    function_lowering.SetLocal(param_id, param_value);
  };

  // The subset of call_param_ids that is already in the order that the LLVM
  // calling convention expects.
  llvm::ArrayRef<SemIR::InstId> sequential_param_ids;
  if (function.return_slot_pattern_id.has_value()) {
    // The LLVM calling convention has the return slot first rather than last.
    // Note that this queries whether there is a return slot at the LLVM level,
    // whereas `function.return_slot_pattern_id.has_value()` queries whether
    // there is a return slot at the SemIR level.
    if (SemIR::ReturnTypeInfo::ForFunction(sem_ir(), function, specific_id)
            .has_return_slot()) {
      lower_param(call_param_ids.back());
    }
    sequential_param_ids = call_param_ids.drop_back();
  } else {
    sequential_param_ids = call_param_ids;
  }

  for (auto param_id : sequential_param_ids) {
    lower_param(param_id);
  }

  auto decl_block_id = SemIR::InstBlockId::None;
  if (function_id == sem_ir().global_ctor_id()) {
    decl_block_id = SemIR::InstBlockId::Empty;
  } else {
    decl_block_id = sem_ir()
                        .insts()
                        .GetAs<SemIR::FunctionDecl>(function.latest_decl_id())
                        .decl_block_id;
  }

  // Lowers the contents of block_id into the corresponding LLVM block,
  // creating it if it doesn't already exist.
  auto lower_block = [&](SemIR::InstBlockId block_id) {
    CARBON_VLOG("Lowering {0}\n", block_id);
    auto* llvm_block = function_lowering.GetBlock(block_id);
    // Keep the LLVM blocks in lexical order.
    llvm_block->moveBefore(llvm_function->end());
    function_lowering.builder().SetInsertPoint(llvm_block);
    function_lowering.LowerBlockContents(block_id);
  };

  lower_block(decl_block_id);

  // If the decl block is empty, reuse it as the first body block. We don't do
  // this when the decl block is non-empty so that any branches back to the
  // first body block don't also re-execute the decl.
  llvm::BasicBlock* block = function_lowering.builder().GetInsertBlock();
  if (block->empty() &&
      function_lowering.TryToReuseBlock(body_block_ids.front(), block)) {
    // Reuse this block as the first block of the function body.
  } else {
    function_lowering.builder().CreateBr(
        function_lowering.GetBlock(body_block_ids.front()));
  }

  // Lower all blocks.
  for (auto block_id : body_block_ids) {
    lower_block(block_id);
  }

  // LLVM requires that the entry block has no predecessors.
  auto* entry_block = &llvm_function->getEntryBlock();
  if (entry_block->hasNPredecessorsOrMore(1)) {
    auto* new_entry_block = llvm::BasicBlock::Create(
        llvm_context(), "entry", llvm_function, entry_block);
    llvm::BranchInst::Create(entry_block, new_entry_block);
  }

  // Emit fingerprint accumulated inside the function context.
  function_lowering.EmitFinalFingerprint();
}

auto FileContext::BuildDISubprogram(const SemIR::Function& function,
                                    const llvm::Function* llvm_function)
    -> llvm::DISubprogram* {
  if (!context().di_compile_unit()) {
    return nullptr;
  }
  auto name = sem_ir().names().GetAsStringIfIdentifier(function.name_id);
  CARBON_CHECK(name, "Unexpected special name for function: {0}",
               function.name_id);
  auto loc = GetLocForDI(function.definition_id);
  // TODO: Add more details here, including real subroutine type (once type
  // information is built), etc.
  return context().di_builder().createFunction(
      context().di_compile_unit(), *name, llvm_function->getName(),
      /*File=*/context().di_builder().createFile(loc.filename, ""),
      /*LineNo=*/loc.line_number,
      context().di_builder().createSubroutineType(
          context().di_builder().getOrCreateTypeArray(std::nullopt)),
      /*ScopeLine=*/0, llvm::DINode::FlagZero,
      llvm::DISubprogram::SPFlagDefinition);
}

// BuildTypeForInst is used to construct types for FileContext::BuildType below.
// Implementations return the LLVM type for the instruction. This first overload
// is the fallback handler for non-type instructions.
template <typename InstT>
  requires(InstT::Kind.is_type() == SemIR::InstIsType::Never)
static auto BuildTypeForInst(FileContext& /*context*/, InstT inst)
    -> llvm::Type* {
  CARBON_FATAL("Cannot use inst as type: {0}", inst);
}

template <typename InstT>
  requires(InstT::Kind.is_symbolic_when_type())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> llvm::Type* {
  // Treat non-monomorphized symbolic types as opaque.
  return llvm::StructType::get(context.llvm_context());
}

static auto BuildTypeForInst(FileContext& context, SemIR::ArrayType inst)
    -> llvm::Type* {
  return llvm::ArrayType::get(
      context.GetType(context.sem_ir().types().GetTypeIdForTypeInstId(
          inst.element_type_inst_id)),
      *context.sem_ir().GetArrayBoundValue(inst.bound_id));
}

static auto BuildTypeForInst(FileContext& /*context*/, SemIR::AutoType inst)
    -> llvm::Type* {
  CARBON_FATAL("Unexpected builtin type in lowering: {0}", inst);
}

static auto BuildTypeForInst(FileContext& context, SemIR::BoolType /*inst*/)
    -> llvm::Type* {
  // TODO: We may want to have different representations for `bool` storage
  // (`i8`) versus for `bool` values (`i1`).
  return llvm::Type::getInt1Ty(context.llvm_context());
}

static auto BuildTypeForInst(FileContext& context, SemIR::ClassType inst)
    -> llvm::Type* {
  auto object_repr_id = context.sem_ir()
                            .classes()
                            .Get(inst.class_id)
                            .GetObjectRepr(context.sem_ir(), inst.specific_id);
  return context.GetType(object_repr_id);
}

static auto BuildTypeForInst(FileContext& context, SemIR::ConstType inst)
    -> llvm::Type* {
  return context.GetType(
      context.sem_ir().types().GetTypeIdForTypeInstId(inst.inner_id));
}

static auto BuildTypeForInst(FileContext& context,
                             SemIR::ImplWitnessAssociatedConstant inst)
    -> llvm::Type* {
  return context.GetType(inst.type_id);
}

static auto BuildTypeForInst(FileContext& /*context*/,
                             SemIR::ErrorInst /*inst*/) -> llvm::Type* {
  // This is a complete type but uses of it should never be lowered.
  return nullptr;
}

static auto BuildTypeForInst(FileContext& context, SemIR::FloatType /*inst*/)
    -> llvm::Type* {
  // TODO: Handle different sizes.
  return llvm::Type::getDoubleTy(context.llvm_context());
}

static auto BuildTypeForInst(FileContext& context, SemIR::IntType inst)
    -> llvm::Type* {
  auto width =
      context.sem_ir().insts().TryGetAs<SemIR::IntValue>(inst.bit_width_id);
  CARBON_CHECK(width, "Can't lower int type with symbolic width");
  return llvm::IntegerType::get(
      context.llvm_context(),
      context.sem_ir().ints().Get(width->int_id).getZExtValue());
}

static auto BuildTypeForInst(FileContext& context,
                             SemIR::LegacyFloatType /*inst*/) -> llvm::Type* {
  return llvm::Type::getDoubleTy(context.llvm_context());
}

static auto BuildTypeForInst(FileContext& context, SemIR::PointerType /*inst*/)
    -> llvm::Type* {
  return llvm::PointerType::get(context.llvm_context(), /*AddressSpace=*/0);
}

static auto BuildTypeForInst(FileContext& /*context*/,
                             SemIR::PatternType /*inst*/) -> llvm::Type* {
  CARBON_FATAL("Unexpected pattern type in lowering");
}

static auto BuildTypeForInst(FileContext& context, SemIR::StructType inst)
    -> llvm::Type* {
  auto fields = context.sem_ir().struct_type_fields().Get(inst.fields_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  subtypes.reserve(fields.size());
  for (auto field : fields) {
    subtypes.push_back(context.GetType(
        context.sem_ir().types().GetTypeIdForTypeInstId(field.type_inst_id)));
  }
  return llvm::StructType::get(context.llvm_context(), subtypes);
}

static auto BuildTypeForInst(FileContext& context, SemIR::TupleType inst)
    -> llvm::Type* {
  // TODO: Investigate special-casing handling of empty tuples so that they
  // can be collectively replaced with LLVM's void, particularly around
  // function returns. LLVM doesn't allow declaring variables with a void
  // type, so that may require significant special casing.
  auto elements = context.sem_ir().inst_blocks().Get(inst.type_elements_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  subtypes.reserve(elements.size());
  for (auto type_id : context.sem_ir().types().GetBlockAsTypeIds(elements)) {
    subtypes.push_back(context.GetType(type_id));
  }
  return llvm::StructType::get(context.llvm_context(), subtypes);
}

static auto BuildTypeForInst(FileContext& context, SemIR::TypeType /*inst*/)
    -> llvm::Type* {
  return context.GetTypeType();
}

static auto BuildTypeForInst(FileContext& context, SemIR::VtableType /*inst*/)
    -> llvm::Type* {
  return llvm::Type::getVoidTy(context.llvm_context());
}

template <typename InstT>
  requires(InstT::Kind.template IsAnyOf<SemIR::SpecificFunctionType,
                                        SemIR::StringType>())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> llvm::Type* {
  // TODO: Decide how we want to represent `StringType`.
  return llvm::PointerType::get(context.llvm_context(), 0);
}

template <typename InstT>
  requires(InstT::Kind
               .template IsAnyOf<SemIR::BoundMethodType, SemIR::IntLiteralType,
                                 SemIR::NamespaceType, SemIR::WitnessType>())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> llvm::Type* {
  // Return an empty struct as a placeholder.
  return llvm::StructType::get(context.llvm_context());
}

template <typename InstT>
  requires(InstT::Kind.template IsAnyOf<
           SemIR::AssociatedEntityType, SemIR::FacetType, SemIR::FunctionType,
           SemIR::FunctionTypeWithSelfType, SemIR::GenericClassType,
           SemIR::GenericInterfaceType, SemIR::InstType,
           SemIR::UnboundElementType, SemIR::WhereExpr>())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> llvm::Type* {
  // Return an empty struct as a placeholder.
  // TODO: Should we model an interface as a witness table, or an associated
  // entity as an index?
  return llvm::StructType::get(context.llvm_context());
}

auto FileContext::BuildType(SemIR::InstId inst_id) -> llvm::Type* {
  // Use overload resolution to select the implementation, producing compile
  // errors when BuildTypeForInst isn't defined for a given instruction.
  CARBON_KIND_SWITCH(sem_ir_->insts().Get(inst_id)) {
#define CARBON_SEM_IR_INST_KIND(Name)     \
  case CARBON_KIND(SemIR::Name inst): {   \
    return BuildTypeForInst(*this, inst); \
  }
#include "toolchain/sem_ir/inst_kind.def"
  }
}

auto FileContext::BuildGlobalVariableDecl(SemIR::VarStorage var_storage)
    -> llvm::GlobalVariable* {
  Mangler m(*this);
  auto mangled_name = m.MangleGlobalVariable(var_storage.pattern_id);
  auto linkage = llvm::GlobalVariable::ExternalLinkage;

  // If the variable doesn't have an externally-visible name, demote it to
  // internal linkage and invent a plausible name that shouldn't collide with
  // any of our real manglings.
  if (mangled_name.empty()) {
    linkage = llvm::GlobalVariable::InternalLinkage;
    if (inst_namer_) {
      mangled_name =
          ("var.anon" + inst_namer_->GetUnscopedNameFor(var_storage.pattern_id))
              .str();
    }
  }

  auto* type = GetType(var_storage.type_id);
  return new llvm::GlobalVariable(llvm_module(), type,
                                  /*isConstant=*/false, linkage,
                                  /*Initializer=*/nullptr, mangled_name);
}

auto FileContext::GetLocForDI(SemIR::InstId inst_id) -> Context::LocForDI {
  return context().GetLocForDI(
      GetAbsoluteNodeId(sem_ir_, SemIR::LocId(inst_id)).back());
}

auto FileContext::BuildVtable(const SemIR::Class& class_info)
    -> llvm::GlobalVariable* {
  // Bail out if this class is not dynamic (this will account for classes that
  // are declared-and-not-defined (including extern declarations) as well).
  if (!class_info.is_dynamic) {
    return nullptr;
  }

  // Vtables can't be generated for generics, only for their specifics - and
  // must be done lazily based on the use of those specifics.
  if (class_info.generic_id != SemIR::GenericId::None) {
    return nullptr;
  }

  Mangler m(*this);
  std::string mangled_name = m.MangleVTable(class_info);

  auto first_owning_decl_loc =
      sem_ir().insts().GetCanonicalLocId(class_info.first_owning_decl_id);
  if (first_owning_decl_loc.kind() == SemIR::LocId::Kind::ImportIRInstId) {
    // Emit a declaration of an imported vtable using a(n opaque) pointer type.
    // This doesn't have to match the definition that appears elsewhere, it'll
    // still get merged correctly.
    auto* gv = new llvm::GlobalVariable(
        llvm_module(),
        llvm::PointerType::get(llvm_context(), /*AddressSpace=*/0),
        /*isConstant=*/true, llvm::GlobalValue::ExternalLinkage, nullptr,
        mangled_name);
    gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    return gv;
  }

  auto canonical_vtable_id =
      sem_ir().constant_values().GetConstantInstId(class_info.vtable_id);

  auto vtable_inst_block =
      sem_ir().inst_blocks().Get(sem_ir()
                                     .insts()
                                     .GetAs<SemIR::Vtable>(canonical_vtable_id)
                                     .virtual_functions_id);

  auto* entry_type = llvm::IntegerType::getInt32Ty(llvm_context());
  auto* table_type = llvm::ArrayType::get(entry_type, vtable_inst_block.size());

  auto* llvm_vtable = new llvm::GlobalVariable(
      llvm_module(), table_type, /*isConstant=*/true,
      llvm::GlobalValue::ExternalLinkage, nullptr, mangled_name);

  auto* i32_type = llvm::IntegerType::getInt32Ty(llvm_context());
  auto* i64_type = llvm::IntegerType::getInt64Ty(llvm_context());
  auto* vtable_const_int =
      llvm::ConstantExpr::getPtrToInt(llvm_vtable, i64_type);

  llvm::SmallVector<llvm::Constant*> vfuncs;
  vfuncs.reserve(vtable_inst_block.size());

  for (auto fn_decl_id : vtable_inst_block) {
    auto fn_decl = GetCalleeFunction(sem_ir(), fn_decl_id);
    vfuncs.push_back(llvm::ConstantExpr::getTrunc(
        llvm::ConstantExpr::getSub(
            llvm::ConstantExpr::getPtrToInt(
                GetOrCreateFunction(fn_decl.function_id,
                                    SemIR::SpecificId::None),
                i64_type),
            vtable_const_int),
        i32_type));
  }

  llvm_vtable->setInitializer(llvm::ConstantArray::get(table_type, vfuncs));
  llvm_vtable->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  return llvm_vtable;
}

}  // namespace Carbon::Lower
