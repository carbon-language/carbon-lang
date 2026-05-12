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
#include "common/pretty_stack_trace_function.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/clang_global_decl.h"
#include "toolchain/lower/constant.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/lower/options.h"
#include "toolchain/lower/specific_coalescer.h"
#include "toolchain/sem_ir/absolute_node_ref.h"
#include "toolchain/sem_ir/diagnostic_loc_converter.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_categories.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/mangler.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/stringify.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

FileContext::FileContext(Context& context, const SemIR::File& sem_ir,
                         const SemIR::InstNamer* inst_namer,
                         llvm::raw_ostream* vlog_stream)
    : context_(&context),
      sem_ir_(&sem_ir),
      inst_namer_(inst_namer),
      vlog_stream_(vlog_stream),
      functions_(LoweredFunctionStore::MakeForOverwrite(sem_ir.functions())),
      specific_functions_(sem_ir.specifics(), std::nullopt),
      types_(LoweredTypeStore::MakeWithExplicitSize(
          sem_ir.constant_values().ConcreteStoreSize(),
          sem_ir.constant_values().GetTypeIdTag(), {nullptr, nullptr})),
      constants_(LoweredConstantStore::MakeWithExplicitSize(
          sem_ir.insts().size(), sem_ir.insts().GetIdTag(), nullptr)),
      lowered_specifics_(sem_ir.generics(),
                         llvm::SmallVector<SemIR::SpecificId>()),
      coalescer_(vlog_stream_, sem_ir.specifics()),
      vtables_(decltype(vtables_)::MakeForOverwrite(sem_ir.vtables())),
      specific_vtables_(sem_ir.specifics(), nullptr) {
  // Initialization that relies on invariants of the class.
  cpp_code_generator_ = cpp_file() ? cpp_file()->GetCodeGenerator() : nullptr;
  CARBON_CHECK(
      !cpp_code_generator_ ||
      (&cpp_code_generator_->GetModule()->getContext() == &llvm_context()));
  CARBON_CHECK(!sem_ir.has_errors(),
               "Generating LLVM IR from invalid SemIR::File is unsupported.");
}

// TODO: Move this to lower.cpp.
auto FileContext::PrepareToLower() -> void {
  // Lower all types that were required to be complete.
  for (auto type_id : sem_ir_->types().complete_types()) {
    if (type_id.index >= 0) {
      types_.Set(type_id,
                 BuildType(*this, sem_ir_->types().GetTypeInstId(type_id)));
    }
  }

  // Lower function declarations.
  for (auto [id, function] : sem_ir_->functions().enumerate()) {
    if (id == sem_ir().global_ctor_id()) {
      // The global constructor is only lowered when we generate its definition.
      // LLVM doesn't allow an internal linkage function to be undefined.
      continue;
    }
    if (function.evaluation_mode == SemIR::Function::EvaluationMode::MustEval) {
      // musteval functions are never lowered.
      continue;
    }
    functions_.Set(id, BuildFunctionDecl(id));
  }

  // TODO: Split vtable declaration creation from definition creation to avoid
  // redundant vtable definitions for imported vtables.
  for (const auto& [id, vtable] : sem_ir_->vtables().enumerate()) {
    const auto& class_info = sem_ir().classes().Get(vtable.class_id);
    // Vtables can't be generated for generics, only for their specifics - and
    // must be done lazily based on the use of those specifics.
    if (!class_info.generic_id.has_value()) {
      vtables_.Set(id, BuildVtable(vtable, SemIR::SpecificId::None));
    }
  }

  // Lower constants.
  LowerConstants(*this, constants_);
}

// TODO: Move this to lower.cpp.
auto FileContext::LowerDefinitions() -> void {
  // Lower global variable definitions.
  // TODO: Storing both a `constants_` array and a separate `global_variables_`
  // map is redundant.
  LowerGlobalVariables(sem_ir().top_inst_block_id());

  // Lower static class variable definitions.
  for (auto class_info : sem_ir().classes().values()) {
    auto inst_block_id = class_info.body_block_id;
    if (inst_block_id.has_value()) {
      LowerGlobalVariables(inst_block_id);
    }
  }

  // Lower function definitions.
  for (auto [id, fn_info] : sem_ir_->functions().enumerate()) {
    // If we created a declaration and the function definition is not imported,
    // build a definition.
    if (functions_.Get(id) && fn_info.definition_id.has_value() &&
        !sem_ir().insts().GetImportSource(fn_info.definition_id).has_value()) {
      BuildFunctionDefinition(id);
    }
  }

  // Append `__global_init` to `llvm::global_ctors` to initialize global
  // variables.
  if (auto global_ctor_id = sem_ir().global_ctor_id();
      global_ctor_id.has_value()) {
    auto llvm_function = BuildFunctionDecl(global_ctor_id);
    functions_.Set(global_ctor_id, llvm_function);
    const auto& global_ctor = sem_ir().functions().Get(global_ctor_id);
    BuildFunctionBody(global_ctor_id, SemIR::SpecificId::None, global_ctor,
                      *this, global_ctor);
    llvm::appendToGlobalCtors(llvm_module(), llvm_function->llvm_function,
                              /*Priority=*/0);
  }
}

auto FileContext::Finalize() -> void {
  if (cpp_code_generator_) {
    // Clang code generation should not actually modify the AST, but isn't
    // const-correct.
    cpp_code_generator_->HandleTranslationUnit(
        const_cast<clang::ASTContext&>(cpp_file()->ast_context()));
  }

  // Find equivalent specifics (from the same generic), replace all uses and
  // remove duplicately lowered function definitions.
  coalescer_.CoalesceEquivalentSpecifics(lowered_specifics_,
                                         specific_functions_);
}

auto FileContext::GetConstant(SemIR::ConstantId const_id,
                              SemIR::InstId use_inst_id) -> llvm::Value* {
  auto const_inst_id = sem_ir().constant_values().GetInstId(const_id);
  auto* const_value = constants_.Get(const_inst_id);

  // For value expressions and initializing expressions, the value produced by
  // a constant instruction is a value representation of the constant. For
  // initializing expressions, `FinishInit` will perform a copy if needed.
  switch (auto cat = SemIR::GetExprCategory(sem_ir(), const_inst_id)) {
    case SemIR::ExprCategory::Value:
    case SemIR::ExprCategory::ReprInitializing:
    case SemIR::ExprCategory::InPlaceInitializing:
      break;

    case SemIR::ExprCategory::DurableRef:
    case SemIR::ExprCategory::EphemeralRef:
      // Constant reference expressions lower to an address.
      return const_value;

    case SemIR::ExprCategory::NotExpr:
    case SemIR::ExprCategory::Error:
    case SemIR::ExprCategory::Pattern:
    case SemIR::ExprCategory::Mixed:
    case SemIR::ExprCategory::RefTagged:
    case SemIR::ExprCategory::Dependent:
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
    if (use_inst_id.has_value()) {
      use_name = inst_namer_->GetUnscopedNameFor(use_inst_id);
    }
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

auto FileContext::GetOrCreateFunctionInfo(
    SemIR::FunctionId function_id, SemIR::SpecificId specific_id,
    FileContext* fallback_file, SemIR::FunctionId fallback_function_id,
    SemIR::SpecificId fallback_specific_id) -> std::optional<FunctionInfo>& {
  // If we have already lowered a declaration of this function, just return it.
  // TODO: If the existing declaration is inexact, and we now have a fallback,
  // we should try again.
  auto& result = GetFunctionInfo(function_id, specific_id);
  if (!result) {
    result = BuildFunctionDecl(function_id, specific_id, fallback_file,
                               fallback_function_id, fallback_specific_id);
  }
  return result;
}

auto FileContext::LowerGlobalVariables(SemIR::InstBlockId inst_block_id)
    -> void {
  for (auto inst_id : sem_ir().inst_blocks().Get(inst_block_id)) {
    // Only `VarStorage` indicates a global variable declaration in the
    // top instruction block.
    if (auto var = sem_ir().insts().TryGetAs<SemIR::VarStorage>(inst_id)) {
      // Get the global variable declaration. We created this when lowering the
      // constant unless the variable is unnamed, in which case we need to
      // create it now.
      llvm::GlobalVariable* llvm_var = nullptr;
      if (auto const_id = sem_ir().constant_values().Get(inst_id);
          const_id.is_constant()) {
        llvm_var = cast<llvm::GlobalVariable>(GetConstant(const_id, inst_id));
      } else {
        // We should never be emitting a definition for a C++ global variable.
        llvm_var = BuildNonCppGlobalVariableDecl(*var);
      }

      // Convert the declaration of this variable into a definition by adding an
      // initializer.
      global_variables_.Insert(inst_id, llvm_var);
      llvm_var->setInitializer(
          llvm::Constant::getNullValue(llvm_var->getValueType()));
    }
  }
}

auto FileContext::HandleReferencedCppFunction(clang::FunctionDecl* cpp_decl)
    -> llvm::Function* {
  // Create the LLVM function (`CodeGenModule::GetOrCreateLLVMFunction()`)
  // so that code generation (`CodeGenModule::EmitGlobal()`) would see this
  // function name (`CodeGenModule::getMangledName()`), and will generate
  // its definition.
  auto* function_address = dyn_cast<llvm::Function>(
      cpp_code_generator_->GetAddrOfGlobal(CreateGlobalDecl(cpp_decl),
                                           /*isForDefinition=*/false));
  CARBON_CHECK(function_address);

  return function_address;
}

auto FileContext::HandleReferencedSpecificFunction(
    SemIR::FunctionId function_id, SemIR::SpecificId specific_id,
    llvm::Type* llvm_type) -> void {
  CARBON_CHECK(specific_id.has_value());

  // Add this specific function to a list of specific functions whose
  // definitions we need to emit.
  // TODO: Don't do this if we know this function is emitted as a
  // non-discardable symbol in the IR for some other file.
  context().AddPendingSpecificFunctionDefinition({.context = this,
                                                  .function_id = function_id,
                                                  .specific_id = specific_id});

  // Create a unique fingerprint for the function type.
  // For now, we compute the function type fingerprint only for specifics,
  // though we might need it for all functions in order to create a canonical
  // fingerprint across translation units.
  coalescer_.CreateTypeFingerprint(specific_id, llvm_type);
}

auto FileContext::GetOrCreateLLVMFunction(
    const FunctionTypeInfo& function_type_info, SemIR::FunctionId function_id,
    SemIR::SpecificId specific_id) -> llvm::Function* {
  // If this is a C++ function, tell Clang that we referenced it.
  if (auto clang_decl_id = sem_ir().functions().Get(function_id).clang_decl_id;
      clang_decl_id.has_value()) {
    CARBON_CHECK(!specific_id.has_value(),
                 "Specific functions cannot have C++ definitions");
    return HandleReferencedCppFunction(
        sem_ir().clang_decls().Get(clang_decl_id).key.decl->getAsFunction());
  }

  SemIR::Mangler m(sem_ir(), context().total_ir_count());
  std::string mangled_name = m.Mangle(function_id, specific_id);
  if (auto* existing = llvm_module().getFunction(mangled_name)) {
    // We might have already lowered this function while lowering a different
    // file. That's OK.
    // TODO: If the prior function was inexact and the new one is not, we should
    // lower this new one and replace the existing function with it.
    // TODO: Check-fail or maybe diagnose if the two LLVM functions are not
    // produced by declarations of the same Carbon function. Name collisions
    // between non-private members of the same library should have been
    // diagnosed by check if detected, but it's not clear that check will
    // always be able to see this problem. In theory, name collisions could
    // also occur due to fingerprint collision.
    return existing;
  }

  // If this is a specific function, we may need to do additional work to
  // emit its definition.
  if (specific_id.has_value()) {
    HandleReferencedSpecificFunction(function_id, specific_id,
                                     function_type_info.type);
  }

  // TODO: For an imported inline function, consider generating an
  // `available_externally` definition.
  auto linkage = llvm::Function::ExternalLinkage;
  if (function_id == sem_ir().global_ctor_id()) {
    // The global constructor name would collide with global constructors for
    // other files in the same package, so use an internal linkage symbol.
    linkage = llvm::Function::InternalLinkage;
  } else if (specific_id.has_value()) {
    // Specific functions are allowed to be duplicated across files.
    // TODO: CoreWitness should have the same behavior; see its use of
    // WeakODRLinkage in BuildFunctionDefinition.
    linkage = llvm::Function::LinkOnceODRLinkage;
  }

  auto* llvm_function = llvm::Function::Create(function_type_info.type, linkage,
                                               mangled_name, llvm_module());
  CARBON_CHECK(llvm_function->getName() == mangled_name,
               "Mangled name collision: {0}", mangled_name);

  // Set up parameters and the return slot.
  for (auto [name_id, arg] : llvm::zip_equal(function_type_info.param_name_ids,
                                             llvm_function->args())) {
    arg.setName(sem_ir().names().GetIRBaseName(name_id));
  }
  if (function_type_info.sret_type != nullptr) {
    auto& return_arg = *llvm_function->args().begin();
    return_arg.addAttr(llvm::Attribute::getWithStructRetType(
        llvm_context(), function_type_info.sret_type));
  }

  return llvm_function;
}

auto FileContext::BuildFunctionDecl(SemIR::FunctionId function_id,
                                    SemIR::SpecificId specific_id,
                                    FileContext* fallback_file,
                                    SemIR::FunctionId fallback_function_id,
                                    SemIR::SpecificId fallback_specific_id)
    -> std::optional<FunctionInfo> {
  const auto& function = sem_ir().functions().Get(function_id);

  // Don't lower generic functions. Note that associated functions in interfaces
  // have `Self` in scope, so are implicitly generic functions.
  if (function.generic_id.has_value() && !specific_id.has_value()) {
    return std::nullopt;
  }

  // Don't lower builtins.
  if (function.builtin_function_kind() != SemIR::BuiltinFunctionKind::None) {
    return std::nullopt;
  }

  // Don't lower C++ functions that use a thunk. We will never reference them
  // directly, and their signatures would not be expected to match the
  // corresponding C++ function anyway.
  if (function.special_function_kind ==
      SemIR::Function::SpecialFunctionKind::HasCppThunk) {
    return std::nullopt;
  }

  // TODO: Consider tracking whether the function has been used, and only
  // lowering it if it's needed.

  FunctionInContext func_infos[] = {
      {this, function_id, specific_id},
      {fallback_file, fallback_function_id, fallback_specific_id}};
  auto function_type_info =
      BuildFunctionTypeInfo(llvm::ArrayRef(func_infos, fallback_file ? 2 : 1));
  auto* llvm_function =
      GetOrCreateLLVMFunction(function_type_info, function_id, specific_id);

  return {{.type = function_type_info.type,
           .di_type = function_type_info.di_type,
           .lowered_param_indices =
               std::move(function_type_info.lowered_param_indices),
           .unused_param_indices =
               std::move(function_type_info.unused_param_indices),
           .llvm_function = llvm_function,
           .inexact = function_type_info.inexact}};
}

// Find the file and function ID describing the definition of a function.
static auto GetFunctionDefinition(const SemIR::File* decl_ir,
                                  SemIR::FunctionId function_id)
    -> std::pair<const SemIR::File*, SemIR::FunctionId> {
  // Find the file containing the definition.
  auto decl_id = decl_ir->functions().Get(function_id).definition_id;
  if (!decl_id.has_value()) {
    // Function is not defined.
    return {nullptr, SemIR::FunctionId::None};
  }

  // Find the function declaration this function was originally imported from.
  while (true) {
    auto import_inst_id = decl_ir->insts().GetImportSource(decl_id);
    if (!import_inst_id.has_value()) {
      break;
    }
    auto import_inst = decl_ir->import_ir_insts().Get(import_inst_id);
    decl_ir = decl_ir->import_irs().Get(import_inst.ir_id()).sem_ir;
    decl_id = import_inst.inst_id();
  }

  auto decl_ir_function_id =
      decl_ir->insts().GetAs<SemIR::FunctionDecl>(decl_id).function_id;
  return {decl_ir, decl_ir_function_id};
}

auto FileContext::BuildFunctionDefinition(SemIR::FunctionId function_id,
                                          SemIR::SpecificId specific_id)
    -> void {
  auto [definition_ir, definition_ir_function_id] =
      GetFunctionDefinition(&sem_ir(), function_id);
  if (!definition_ir) {
    // Function is probably defined in another file; not an error.
    return;
  }

  const auto& definition_function =
      definition_ir->functions().Get(definition_ir_function_id);
  BuildFunctionBody(
      function_id, specific_id, sem_ir().functions().Get(function_id),
      context().GetFileContext(definition_ir), definition_function);
}

auto FileContext::BuildFunctionBody(SemIR::FunctionId function_id,
                                    SemIR::SpecificId specific_id,
                                    const SemIR::Function& declaration_function,
                                    FileContext& definition_context,
                                    const SemIR::Function& definition_function)
    -> void {
  // On crash, report the function we were lowering.
  PrettyStackTraceFunction stack_trace_entry([&](llvm::raw_ostream& output) {
    SemIR::DiagnosticLocConverter converter(
        &context().tree_and_subtrees_getters(), &sem_ir());
    auto converted =
        converter.Convert(SemIR::LocId(declaration_function.definition_id),
                          /*token_only=*/false);
    converted.loc.FormatLocation(output);
    output << "Lowering function ";
    if (specific_id.has_value()) {
      output << SemIR::StringifySpecific(sem_ir(), specific_id);
    } else {
      output << SemIR::StringifyConstantInst(
          sem_ir(), declaration_function.definition_id);
    }
    output << "\n";
    // Crash output has a tab indent; try to indent slightly past that.
    converted.loc.FormatSnippet(output, /*indent=*/10);
  });

  // Note that `definition_function` is potentially from a different SemIR::File
  // than the one that this file context represents. Any lowering done for
  // values derived from `definition_function` should use `definition_context`
  // instead of our context.
  const auto& definition_ir = definition_context.sem_ir();

  auto function_info = GetFunctionInfo(function_id, specific_id);
  CARBON_CHECK(function_info && function_info->llvm_function,
               "Attempting to define function that was not declared");
  CARBON_CHECK(!function_info->inexact,
               "Attempting to emit definition of inexact function: {0}",
               *function_info->llvm_function);

  // TODO: Build CoreWitness functions when they're called instead of when
  // they're defined. That should allow LinkOnceODRLinkage.
  if (declaration_function.special_function_kind ==
      SemIR::Function::SpecialFunctionKind::CoreWitness) {
    function_info->llvm_function->setLinkage(llvm::Function::WeakODRLinkage);
  }

  const auto& body_block_ids = definition_function.body_block_ids;
  CARBON_DCHECK(!body_block_ids.empty(),
                "No function body blocks found during lowering.");

  // Store which specifics were already lowered (with definitions) for each
  // generic.
  if (declaration_function.generic_id.has_value() && specific_id.has_value()) {
    // TODO: We should track this in the definition context instead so that we
    // can deduplicate specifics from different files.
    AddLoweredSpecificForGeneric(declaration_function.generic_id, specific_id);
  }

  // Set attributes on the function definition.
  {
    llvm::AttrBuilder attr_builder(llvm_context());
    attr_builder.addAttribute(llvm::Attribute::NoUnwind);

    // TODO: We should take the opt level from the SemIR file; it might not be
    // the same for all files in a compilation.
    if (context().opt_level() == Lower::OptimizationLevel::None) {
      // --optimize=none disables all optimizations for this function.
      attr_builder.addAttribute(llvm::Attribute::OptimizeNone);
      attr_builder.addAttribute(llvm::Attribute::NoInline);
    } else {
      // Otherwise, always inline thunks.
      if (definition_function.special_function_kind ==
          SemIR::Function::SpecialFunctionKind::Thunk) {
        attr_builder.addAttribute(llvm::Attribute::AlwaysInline);
      }

      // Convert --optimize=size into optsize and minsize.
      if (context().opt_level() == Lower::OptimizationLevel::Size) {
        attr_builder.addAttribute(llvm::Attribute::OptimizeForSize);
        attr_builder.addAttribute(llvm::Attribute::MinSize);
      }

      // TODO: Should we generate an InlineHint for some functions? Perhaps for
      // those defined in the API file?
    }
    function_info->llvm_function->addFnAttrs(attr_builder);
  }

  auto* subprogram = BuildDISubprogram(declaration_function, *function_info);
  FunctionContext function_lowering(
      definition_context, function_info->llvm_function, *this, function_id,
      specific_id, coalescer_.InitializeFingerprintForSpecific(specific_id),
      subprogram, vlog_stream_);

  auto call_param_ids = definition_ir.inst_blocks().GetOrEmpty(
      definition_function.call_params_id);

  // Add local variables for the parameters.
  for (auto [llvm_index, index] :
       llvm::enumerate(function_info->lowered_param_indices)) {
    function_lowering.SetLocal(
        call_param_ids[index.index],
        function_info->llvm_function->getArg(llvm_index));
  }

  // Add local variables for the SemIR parameters that aren't LLVM parameters.
  // These shouldn't actually be used, so they're set to poison values.
  for (auto [llvm_index, index] :
       llvm::enumerate(function_info->unused_param_indices)) {
    auto param_id = call_param_ids[index.index];
    function_lowering.SetLocal(
        param_id,
        llvm::PoisonValue::get(function_lowering.GetTypeOfInst(param_id)));
  }

  auto decl_block_id = SemIR::InstBlockId::None;
  if (function_id == sem_ir().global_ctor_id()) {
    decl_block_id = SemIR::InstBlockId::Empty;
  } else {
    decl_block_id =
        definition_ir.insts()
            .GetAs<SemIR::FunctionDecl>(definition_function.latest_decl_id())
            .decl_block_id;
  }

  // Lowers the contents of decl_block_id into the corresponding LLVM block,
  // creating it if it doesn't already exist.
  auto lower_block = [&](SemIR::InstBlockId block_id) {
    CARBON_VLOG("Lowering {0}\n", block_id);
    auto* llvm_block = function_lowering.GetBlock(block_id);
    // Keep the LLVM blocks in lexical order.
    llvm_block->moveBefore(function_info->llvm_function->end());
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
  auto* entry_block = &function_info->llvm_function->getEntryBlock();
  if (entry_block->hasNPredecessorsOrMore(1)) {
    auto* new_entry_block = llvm::BasicBlock::Create(
        llvm_context(), "entry", function_info->llvm_function, entry_block);
    llvm::UncondBrInst::Create(entry_block, new_entry_block);
  }

  // Emit fingerprint accumulated inside the function context.
  function_lowering.EmitFinalFingerprint();
  context().di_builder().finalizeSubprogram(subprogram);
}

auto FileContext::BuildDISubprogram(const SemIR::Function& function,
                                    const FunctionInfo& function_info)
    -> llvm::DISubprogram* {
  if (!context().di_compile_unit()) {
    return nullptr;
  }
  auto name = sem_ir().names().GetAsStringIfIdentifier(function.name_id);
  CARBON_CHECK(name, "Unexpected special name for function: {0}",
               function.name_id);
  auto loc = GetLocForDI(function.definition_id);
  llvm::DISubroutineType* subroutine_type = function_info.di_type;
  auto* subprogram = context().di_builder().createFunction(
      context().di_compile_unit(), *name,
      function_info.llvm_function->getName(),
      /*File=*/context().di_builder().createFile(loc.filename, ""),
      /*LineNo=*/loc.line_number, subroutine_type,
      /*ScopeLine=*/0, llvm::DINode::FlagZero,
      llvm::DISubprogram::SPFlagDefinition);
  // Add a variable for each parameter, as that is where DWARF debug information
  // comes from.
  // TODO: this doesn't declare a variable for the output parameter. Is that
  // what we want?
  for (auto [argument_number, type] :
       llvm::enumerate(llvm::drop_begin(subroutine_type->getTypeArray()))) {
    context().di_builder().createParameterVariable(
        subprogram, "", argument_number + 1, nullptr, 0, type,
        /*AlwaysPreserve=*/true);
  }
  return subprogram;
}

auto FileContext::BuildGlobalVariableDecl(SemIR::VarStorage var_storage)
    -> llvm::Constant* {
  auto var_name_id =
      SemIR::GetFirstBindingNameFromPatternId(sem_ir(), var_storage.pattern_id);
  if (auto cpp_global_var_id =
          sem_ir().cpp_global_vars().Lookup({.entity_name_id = var_name_id});
      cpp_global_var_id.has_value()) {
    SemIR::ClangDeclId clang_decl_id =
        sem_ir().cpp_global_vars().Get(cpp_global_var_id).clang_decl_id;
    CARBON_CHECK(clang_decl_id.has_value(),
                 "CppGlobalVar should have a clang_decl_id");
    return cpp_code_generator_->GetAddrOfGlobal(
        cast<clang::VarDecl>(
            sem_ir().clang_decls().Get(clang_decl_id).key.decl),
        /*isForDefinition=*/false);
  }

  return BuildNonCppGlobalVariableDecl(var_storage);
}

auto FileContext::BuildNonCppGlobalVariableDecl(SemIR::VarStorage var_storage)
    -> llvm::GlobalVariable* {
  SemIR::Mangler m(sem_ir(), context().total_ir_count());
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
  auto abs_node_ref = GetAbsoluteNodeRef(sem_ir_, SemIR::LocId(inst_id)).back();
  return context().GetLocForDI(abs_node_ref);
}

auto FileContext::BuildVtable(const SemIR::Vtable& vtable,
                              SemIR::SpecificId specific_id)
    -> llvm::GlobalVariable* {
  if (!vtable.carbon_native_vtable) {
    return nullptr;
  }
  const auto& class_info = sem_ir().classes().Get(vtable.class_id);

  SemIR::Mangler m(sem_ir(), context().total_ir_count());
  std::string mangled_name = m.MangleVTable(class_info, specific_id);

  if (sem_ir()
          .insts()
          .GetImportSource(class_info.first_owning_decl_id)
          .has_value()) {
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

  auto vtable_inst_block =
      sem_ir().inst_blocks().Get(vtable.virtual_functions_id);

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
    auto [_1, _2, fn_id, fn_specific_id] =
        DecomposeVirtualFunction(sem_ir(), fn_decl_id, specific_id);

    vfuncs.push_back(llvm::ConstantExpr::getTrunc(
        llvm::ConstantExpr::getSub(
            llvm::ConstantExpr::getPtrToInt(
                GetOrCreateFunctionInfo(fn_id, fn_specific_id)->llvm_function,
                i64_type),
            vtable_const_int),
        i32_type));
  }

  llvm_vtable->setInitializer(llvm::ConstantArray::get(table_type, vfuncs));
  llvm_vtable->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  return llvm_vtable;
}

}  // namespace Carbon::Lower
