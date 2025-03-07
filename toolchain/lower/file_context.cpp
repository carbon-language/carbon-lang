// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/file_context.h"

#include "common/check.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/constant.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/lower/mangler.h"
#include "toolchain/sem_ir/absolute_node_id.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

FileContext::FileContext(
    llvm::LLVMContext& llvm_context,
    std::optional<llvm::ArrayRef<Parse::GetTreeAndSubtreesFn>>
        tree_and_subtrees_getters_for_debug_info,
    llvm::StringRef module_name, const SemIR::File& sem_ir,
    const SemIR::InstNamer* inst_namer, llvm::raw_ostream* vlog_stream)
    : llvm_context_(&llvm_context),
      llvm_module_(std::make_unique<llvm::Module>(module_name, llvm_context)),
      di_builder_(*llvm_module_),
      di_compile_unit_(
          tree_and_subtrees_getters_for_debug_info
              ? BuildDICompileUnit(module_name, *llvm_module_, di_builder_)
              : nullptr),
      tree_and_subtrees_getters_for_debug_info_(
          tree_and_subtrees_getters_for_debug_info),
      sem_ir_(&sem_ir),
      inst_namer_(inst_namer),
      vlog_stream_(vlog_stream) {
  CARBON_CHECK(!sem_ir.has_errors(),
               "Generating LLVM IR from invalid SemIR::File is unsupported.");
}

// TODO: Move this to lower.cpp.
auto FileContext::Run() -> std::unique_ptr<llvm::Module> {
  CARBON_CHECK(llvm_module_, "Run can only be called once.");

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

  // Lower global variable declarations.
  for (auto inst_id :
       sem_ir().inst_blocks().Get(sem_ir().top_inst_block_id())) {
    // Only `VarStorage` indicates a global variable declaration in the
    // top instruction block.
    if (auto var = sem_ir().insts().TryGetAs<SemIR::VarStorage>(inst_id)) {
      global_variables_.Insert(inst_id, BuildGlobalVariableDecl(*var));
    }
  }

  // Lower constants.
  constants_.resize(sem_ir_->insts().size());
  LowerConstants(*this, constants_);

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

  return std::move(llvm_module_);
}

auto FileContext::BuildDICompileUnit(llvm::StringRef module_name,
                                     llvm::Module& llvm_module,
                                     llvm::DIBuilder& di_builder)
    -> llvm::DICompileUnit* {
  llvm_module.addModuleFlag(llvm::Module::Max, "Dwarf Version", 5);
  llvm_module.addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                            llvm::DEBUG_METADATA_VERSION);
  // TODO: Include directory path in the compile_unit_file.
  llvm::DIFile* compile_unit_file = di_builder.createFile(module_name, "");
  // TODO: Introduce a new language code for Carbon. C works well for now since
  // it's something debuggers will already know/have support for at least.
  // Probably have to bump to C++ at some point for virtual functions,
  // templates, etc.
  return di_builder.createCompileUnit(llvm::dwarf::DW_LANG_C, compile_unit_file,
                                      "carbon",
                                      /*isOptimized=*/false, /*Flags=*/"",
                                      /*RV=*/0);
}

auto FileContext::GetGlobal(SemIR::InstId inst_id,
                            SemIR::SpecificId specific_id) -> llvm::Value* {
  auto const_id = GetConstantValueInSpecific(sem_ir(), specific_id, inst_id);
  if (const_id.is_concrete()) {
    auto const_inst_id = sem_ir().constant_values().GetInstId(const_id);

    // For value expressions and initializing expressions, the value produced by
    // a constant instruction is a value representation of the constant. For
    // initializing expressions, `FinishInit` will perform a copy if needed.
    // TODO: Handle reference expression constants.
    auto* const_value = constants_[const_inst_id.index];

    // If we want a pointer to the constant, materialize a global to hold it.
    // TODO: We could reuse the same global if the constant is used more than
    // once.
    auto value_rep = SemIR::ValueRepr::ForType(
        sem_ir(), sem_ir().insts().Get(const_inst_id).type_id());
    if (value_rep.kind == SemIR::ValueRepr::Pointer) {
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

      return new llvm::GlobalVariable(
          llvm_module(), GetType(sem_ir().GetPointeeType(value_rep.type_id)),
          /*isConstant=*/true, llvm::GlobalVariable::InternalLinkage,
          const_value, const_name + sep + use_name);
    }

    // Otherwise, we can use the constant value directly.
    return const_value;
  }

  CARBON_FATAL("Missing value: {0} {1} {2}", inst_id, specific_id,
               sem_ir().insts().Get(inst_id));
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

  // TODO nit: add is_symbolic() to type_id to forward to
  // type_id.AsConstantId().is_symbolic(). Update call below too.
  auto get_llvm_type = [&](SemIR::TypeId type_id) -> llvm::Type* {
    if (!type_id.has_value()) {
      return nullptr;
    }
    return GetType(SemIR::GetTypeInSpecific(sem_ir(), specific_id, type_id));
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
        llvm::PointerType::get(return_type, /*AddressSpace=*/0));
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
    auto param_type_id = SemIR::GetTypeInSpecific(
        sem_ir(), specific_id, param_pattern_info->inst.type_id);
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

  Mangler m(*this);
  std::string mangled_name = m.Mangle(function_id, specific_id);

  auto* llvm_function = llvm::Function::Create(function_type_info.type,
                                               llvm::Function::ExternalLinkage,
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
  if (body_block_ids.empty()) {
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
  CARBON_DCHECK(!body_block_ids.empty(),
                "No function body blocks found during lowering.");

  FunctionContext function_lowering(*this, llvm_function, specific_id,
                                    BuildDISubprogram(function, llvm_function),
                                    vlog_stream_);

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
          SemIR::GetTypeInSpecific(sem_ir(), specific_id, param_inst.type_id)));
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
}

auto FileContext::BuildDISubprogram(const SemIR::Function& function,
                                    const llvm::Function* llvm_function)
    -> llvm::DISubprogram* {
  if (!di_compile_unit_) {
    return nullptr;
  }
  auto name = sem_ir().names().GetAsStringIfIdentifier(function.name_id);
  CARBON_CHECK(name, "Unexpected special name for function: {0}",
               function.name_id);
  auto loc = GetLocForDI(function.definition_id);
  // TODO: Add more details here, including real subroutine type (once type
  // information is built), etc.
  return di_builder_.createFunction(
      di_compile_unit_, *name, llvm_function->getName(),
      /*File=*/di_builder_.createFile(loc.filename, ""),
      /*LineNo=*/loc.line_number,
      di_builder_.createSubroutineType(
          di_builder_.getOrCreateTypeArray(std::nullopt)),
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
  requires(InstT::Kind.constant_kind() ==
               SemIR::InstConstantKind::SymbolicOnly &&
           InstT::Kind.is_type() != SemIR::InstIsType::Never)
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> llvm::Type* {
  // Treat non-monomorphized symbolic types as opaque.
  return llvm::StructType::get(context.llvm_context());
}

static auto BuildTypeForInst(FileContext& context, SemIR::ArrayType inst)
    -> llvm::Type* {
  return llvm::ArrayType::get(
      context.GetType(inst.element_type_id),
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
  return context.GetType(inst.inner_id);
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

static auto BuildTypeForInst(FileContext& context, SemIR::StructType inst)
    -> llvm::Type* {
  auto fields = context.sem_ir().struct_type_fields().Get(inst.fields_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  subtypes.reserve(fields.size());
  for (auto field : fields) {
    subtypes.push_back(context.GetType(field.type_id));
  }
  return llvm::StructType::get(context.llvm_context(), subtypes);
}

static auto BuildTypeForInst(FileContext& context, SemIR::TupleType inst)
    -> llvm::Type* {
  // TODO: Investigate special-casing handling of empty tuples so that they
  // can be collectively replaced with LLVM's void, particularly around
  // function returns. LLVM doesn't allow declaring variables with a void
  // type, so that may require significant special casing.
  auto elements = context.sem_ir().type_blocks().Get(inst.elements_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  subtypes.reserve(elements.size());
  for (auto element_id : elements) {
    subtypes.push_back(context.GetType(element_id));
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
           SemIR::GenericInterfaceType, SemIR::UnboundElementType,
           SemIR::WhereExpr>())
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
  // TODO: Mangle name.
  auto mangled_name =
      *sem_ir().names().GetAsStringIfIdentifier(var_storage.pretty_name_id);
  auto* type = GetType(var_storage.type_id);
  return new llvm::GlobalVariable(
      llvm_module(), type,
      /*isConstant=*/false, llvm::GlobalVariable::InternalLinkage,
      llvm::Constant::getNullValue(type), mangled_name);
}

auto FileContext::GetLocForDI(SemIR::InstId inst_id) -> LocForDI {
  SemIR::AbsoluteNodeId resolved = GetAbsoluteNodeId(sem_ir_, inst_id).back();
  const auto& tree_and_subtrees =
      (*tree_and_subtrees_getters_for_debug_info_)[resolved.check_ir_id
                                                       .index]();
  const auto& tokens = tree_and_subtrees.tree().tokens();

  if (resolved.node_id.has_value()) {
    auto token = tree_and_subtrees.GetSubtreeTokenRange(resolved.node_id).begin;
    return {.filename = tokens.source().filename(),
            .line_number = tokens.GetLineNumber(token),
            .column_number = tokens.GetColumnNumber(token)};
  } else {
    return {.filename = tokens.source().filename(),
            .line_number = 0,
            .column_number = 0};
  }
}

}  // namespace Carbon::Lower
