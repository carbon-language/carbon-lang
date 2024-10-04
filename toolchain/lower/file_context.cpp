// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/file_context.h"

#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/constant.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/lower/mangler.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

FileContext::FileContext(llvm::LLVMContext& llvm_context,
                         bool include_debug_info,
                         const Check::SemIRDiagnosticConverter& converter,
                         llvm::StringRef module_name, const SemIR::File& sem_ir,
                         const SemIR::InstNamer* inst_namer,
                         llvm::raw_ostream* vlog_stream)
    : llvm_context_(&llvm_context),
      llvm_module_(std::make_unique<llvm::Module>(module_name, llvm_context)),
      di_builder_(*llvm_module_),
      di_compile_unit_(
          include_debug_info
              ? BuildDICompileUnit(module_name, *llvm_module_, di_builder_)
              : nullptr),
      converter_(converter),
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
  for (auto i : llvm::seq(sem_ir_->functions().size())) {
    functions_[i] = BuildFunctionDecl(SemIR::FunctionId(i));
  }

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
  for (auto i : llvm::seq(sem_ir_->functions().size())) {
    BuildFunctionDefinition(SemIR::FunctionId(i));
  }
  // Append `__global_init` to `llvm::global_ctors` to initialize global
  // variables.
  if (sem_ir().global_ctor_id().is_valid()) {
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
  // FIXME: Include directory path in the compile_unit_file.
  llvm::DIFile* compile_unit_file = di_builder.createFile(module_name, "");
  // FIXME: Introduce a new language code for Carbon. C works well for now since
  // it's something debuggers will already know/have support for at least.
  // Probably have to bump to C++ at some point for virtual functions,
  // templates, etc.
  return di_builder.createCompileUnit(llvm::dwarf::DW_LANG_C, compile_unit_file,
                                      "carbon",
                                      /*isOptimized=*/false, /*Flags=*/"",
                                      /*RV=*/0);
}

auto FileContext::GetGlobal(SemIR::InstId inst_id) -> llvm::Value* {
  auto inst = sem_ir().insts().Get(inst_id);

  auto const_id = sem_ir().constant_values().Get(inst_id);
  if (const_id.is_template()) {
    auto const_inst_id = sem_ir().constant_values().GetInstId(const_id);

    // For value expressions and initializing expressions, the value produced by
    // a constant instruction is a value representation of the constant. For
    // initializing expressions, `FinishInit` will perform a copy if needed.
    // TODO: Handle reference expression constants.
    auto* const_value = constants_[const_inst_id.index];

    // If we want a pointer to the constant, materialize a global to hold it.
    // TODO: We could reuse the same global if the constant is used more than
    // once.
    auto value_rep = SemIR::ValueRepr::ForType(sem_ir(), inst.type_id());
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

  // TODO: For generics, handle references to symbolic constants.

  CARBON_FATAL("Missing value: {0} {1}", inst_id,
               sem_ir().insts().Get(inst_id));
}

auto FileContext::BuildFunctionDecl(SemIR::FunctionId function_id)
    -> llvm::Function* {
  const auto& function = sem_ir().functions().Get(function_id);

  // Don't lower generic functions or associated functions.
  // TODO: Associated functions have `Self` in scope so should be treated as
  // generic functions.
  if (function.generic_id.is_valid() ||
      sem_ir().insts().Is<SemIR::InterfaceDecl>(
          sem_ir().name_scopes().Get(function.parent_scope_id).inst_id)) {
    return nullptr;
  }

  // Don't lower builtins.
  if (function.builtin_function_kind != SemIR::BuiltinFunctionKind::None) {
    return nullptr;
  }

  // TODO: Consider tracking whether the function has been used, and only
  // lowering it if it's needed.

  // TODO: Pass in a specific ID for generic functions.
  const auto specific_id = SemIR::SpecificId::Invalid;

  const auto return_info =
      SemIR::ReturnTypeInfo::ForFunction(sem_ir(), function, specific_id);
  CARBON_CHECK(return_info.is_valid(), "Should not lower invalid functions.");

  auto implicit_param_refs =
      sem_ir().inst_blocks().GetOrEmpty(function.implicit_param_refs_id);
  // TODO: Include parameters corresponding to positional parameters.
  auto param_refs = sem_ir().inst_blocks().GetOrEmpty(function.param_refs_id);

  auto* return_type =
      return_info.type_id.is_valid() ? GetType(return_info.type_id) : nullptr;

  llvm::SmallVector<llvm::Type*> param_types;
  // TODO: Consider either storing `param_inst_ids` somewhere so that we can
  // reuse it from `BuildFunctionDefinition` and when building calls, or factor
  // out a mechanism to compute the mapping between parameters and arguments on
  // demand.
  llvm::SmallVector<SemIR::InstId> param_inst_ids;
  auto max_llvm_params = (return_info.has_return_slot() ? 1 : 0) +
                         implicit_param_refs.size() + param_refs.size();
  param_types.reserve(max_llvm_params);
  param_inst_ids.reserve(max_llvm_params);
  if (return_info.has_return_slot()) {
    param_types.push_back(return_type->getPointerTo());
    param_inst_ids.push_back(function.return_storage_id);
  }
  for (auto param_ref_id :
       llvm::concat<const SemIR::InstId>(implicit_param_refs, param_refs)) {
    auto param_info =
        SemIR::Function::GetParamFromParamRefId(sem_ir(), param_ref_id);
    if (!param_info.inst.runtime_index.is_valid()) {
      continue;
    }
    switch (auto value_rep =
                SemIR::ValueRepr::ForType(sem_ir(), param_info.inst.type_id);
            value_rep.kind) {
      case SemIR::ValueRepr::Unknown:
        CARBON_FATAL("Incomplete parameter type lowering function declaration");
      case SemIR::ValueRepr::None:
        break;
      case SemIR::ValueRepr::Copy:
      case SemIR::ValueRepr::Custom:
      case SemIR::ValueRepr::Pointer:
        param_types.push_back(GetType(value_rep.type_id));
        param_inst_ids.push_back(param_ref_id);
        break;
    }
  }

  // Compute the return type to use for the LLVM function. If the initializing
  // representation doesn't produce a value, set the return type to void.
  // TODO: For the `Run` entry point, remap return type to i32 if it doesn't
  // return a value.
  llvm::Type* function_return_type =
      return_info.init_repr.kind == SemIR::InitRepr::ByCopy
          ? return_type
          : llvm::Type::getVoidTy(llvm_context());

  Mangler m(*this);
  std::string mangled_name = m.Mangle(function_id);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      function_return_type, param_types, /*isVarArg=*/false);
  auto* llvm_function =
      llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                             mangled_name, llvm_module());

  CARBON_CHECK(llvm_function->getName() == mangled_name,
               "Mangled name collision: {0}", mangled_name);

  // Set up parameters and the return slot.
  for (auto [inst_id, arg] :
       llvm::zip_equal(param_inst_ids, llvm_function->args())) {
    auto name_id = SemIR::NameId::Invalid;
    if (inst_id == function.return_storage_id) {
      name_id = SemIR::NameId::ReturnSlot;
      arg.addAttr(
          llvm::Attribute::getWithStructRetType(llvm_context(), return_type));
    } else {
      name_id = SemIR::Function::GetParamFromParamRefId(sem_ir(), inst_id)
                    .GetNameId(sem_ir());
    }
    arg.setName(sem_ir().names().GetIRBaseName(name_id));
  }

  return llvm_function;
}

auto FileContext::BuildFunctionDefinition(SemIR::FunctionId function_id)
    -> void {
  const auto& function = sem_ir().functions().Get(function_id);
  const auto& body_block_ids = function.body_block_ids;
  if (body_block_ids.empty()) {
    // Function is probably defined in another file; not an error.
    return;
  }

  llvm::Function* llvm_function = GetFunction(function_id);
  if (!llvm_function) {
    // We chose not to lower this function at all, for example because it's a
    // generic function.
    return;
  }

  FunctionContext function_lowering(*this, llvm_function,
                                    BuildDISubprogram(function, llvm_function),
                                    vlog_stream_);

  // TODO: Pass in a specific ID for generic functions.
  const auto specific_id = SemIR::SpecificId::Invalid;

  // Add parameters to locals.
  // TODO: This duplicates the mapping between sem_ir instructions and LLVM
  // function parameters that was already computed in BuildFunctionDecl.
  // We should only do that once.
  auto implicit_param_refs =
      sem_ir().inst_blocks().GetOrEmpty(function.implicit_param_refs_id);
  auto param_refs = sem_ir().inst_blocks().GetOrEmpty(function.param_refs_id);
  int param_index = 0;
  if (SemIR::ReturnTypeInfo::ForFunction(sem_ir(), function, specific_id)
          .has_return_slot()) {
    function_lowering.SetLocal(function.return_storage_id,
                               llvm_function->getArg(param_index));
    ++param_index;
  }
  for (auto param_ref_id :
       llvm::concat<const SemIR::InstId>(implicit_param_refs, param_refs)) {
    auto param_info =
        SemIR::Function::GetParamFromParamRefId(sem_ir(), param_ref_id);
    if (!param_info.inst.runtime_index.is_valid()) {
      continue;
    }

    // Get the value of the parameter from the function argument.
    auto param_type_id = param_info.inst.type_id;
    llvm::Value* param_value = llvm::PoisonValue::get(GetType(param_type_id));
    if (SemIR::ValueRepr::ForType(sem_ir(), param_type_id).kind !=
        SemIR::ValueRepr::None) {
      param_value = llvm_function->getArg(param_index);
      ++param_index;
    }

    // The value of the parameter is the value of the argument.
    function_lowering.SetLocal(param_info.inst_id, param_value);

    // Match the portion of the pattern corresponding to the parameter against
    // the parameter value. For now this is always a single name binding,
    // possibly wrapped in `addr`.
    //
    // TODO: Support general patterns here.
    auto bind_name_id = param_ref_id;
    if (auto addr =
            sem_ir().insts().TryGetAs<SemIR::AddrPattern>(param_ref_id)) {
      bind_name_id = addr->inner_id;
    }
    auto bind_name = sem_ir().insts().Get(bind_name_id);
    CARBON_CHECK(bind_name.Is<SemIR::BindName>());
    function_lowering.SetLocal(bind_name_id, param_value);
  }

  // Lower all blocks.
  for (auto block_id : body_block_ids) {
    CARBON_VLOG("Lowering {0}\n", block_id);
    auto* llvm_block = function_lowering.GetBlock(block_id);
    // Keep the LLVM blocks in lexical order.
    llvm_block->moveBefore(llvm_function->end());
    function_lowering.builder().SetInsertPoint(llvm_block);
    function_lowering.LowerBlock(block_id);
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
  // FIXME: Add more details here, including real subroutine type (once type
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

static auto BuildTypeForInst(FileContext& context, SemIR::ArrayType inst)
    -> llvm::Type* {
  return llvm::ArrayType::get(
      context.GetType(inst.element_type_id),
      context.sem_ir().GetArrayBoundValue(inst.bound_id));
}

static auto BuildTypeForInst(FileContext& context, SemIR::BuiltinInst inst)
    -> llvm::Type* {
  switch (inst.builtin_inst_kind) {
    case SemIR::BuiltinInstKind::Invalid:
      CARBON_FATAL("Unexpected builtin type in lowering.");
    case SemIR::BuiltinInstKind::Error:
      // This is a complete type but uses of it should never be lowered.
      return nullptr;
    case SemIR::BuiltinInstKind::TypeType:
      return context.GetTypeType();
    case SemIR::BuiltinInstKind::FloatType:
      return llvm::Type::getDoubleTy(context.llvm_context());
    case SemIR::BuiltinInstKind::IntType:
      return llvm::Type::getInt32Ty(context.llvm_context());
    case SemIR::BuiltinInstKind::BoolType:
      // TODO: We may want to have different representations for `bool`
      // storage
      // (`i8`) versus for `bool` values (`i1`).
      return llvm::Type::getInt1Ty(context.llvm_context());
    case SemIR::BuiltinInstKind::StringType:
      // TODO: Decide how we want to represent `StringType`.
      return llvm::PointerType::get(context.llvm_context(), 0);
    case SemIR::BuiltinInstKind::BoundMethodType:
    case SemIR::BuiltinInstKind::NamespaceType:
    case SemIR::BuiltinInstKind::WitnessType:
      // Return an empty struct as a placeholder.
      return llvm::StructType::get(context.llvm_context());
  }
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

static auto BuildTypeForInst(FileContext& context, SemIR::FloatType /*inst*/)
    -> llvm::Type* {
  // TODO: Handle different sizes.
  return llvm::Type::getDoubleTy(context.llvm_context());
}

static auto BuildTypeForInst(FileContext& context, SemIR::IntType inst)
    -> llvm::Type* {
  auto width =
      context.sem_ir().insts().TryGetAs<SemIR::IntLiteral>(inst.bit_width_id);
  CARBON_CHECK(width, "Can't lower int type with symbolic width");
  return llvm::IntegerType::get(
      context.llvm_context(),
      context.sem_ir().ints().Get(width->int_id).getZExtValue());
}

static auto BuildTypeForInst(FileContext& context, SemIR::PointerType /*inst*/)
    -> llvm::Type* {
  return llvm::PointerType::get(context.llvm_context(), /*AddressSpace=*/0);
}

static auto BuildTypeForInst(FileContext& context, SemIR::StructType inst)
    -> llvm::Type* {
  auto fields = context.sem_ir().inst_blocks().Get(inst.fields_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  subtypes.reserve(fields.size());
  for (auto field_id : fields) {
    auto field =
        context.sem_ir().insts().GetAs<SemIR::StructTypeField>(field_id);
    subtypes.push_back(context.GetType(field.field_type_id));
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

template <typename InstT>
  requires(InstT::Kind.template IsAnyOf<
           SemIR::AssociatedEntityType, SemIR::FunctionType,
           SemIR::GenericClassType, SemIR::GenericInterfaceType,
           SemIR::InterfaceType, SemIR::UnboundElementType, SemIR::WhereExpr>())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> llvm::Type* {
  // Return an empty struct as a placeholder.
  // TODO: Should we model an interface as a witness table, or an associated
  // entity as an index?
  return llvm::StructType::get(context.llvm_context());
}

// Treat non-monomorphized symbolic types as opaque.
template <typename InstT>
  requires(InstT::Kind.template IsAnyOf<SemIR::BindSymbolicName,
                                        SemIR::InterfaceWitnessAccess>())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> llvm::Type* {
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
      *sem_ir().names().GetAsStringIfIdentifier(var_storage.name_id);
  auto* type =
      var_storage.type_id.is_valid() ? GetType(var_storage.type_id) : nullptr;
  return new llvm::GlobalVariable(llvm_module(), type,
                                  /*isConstant=*/false,
                                  llvm::GlobalVariable::InternalLinkage,
                                  /*Initializer=*/nullptr, mangled_name);
}

auto FileContext::GetLocForDI(SemIR::InstId inst_id) -> LocForDI {
  auto diag_loc = converter_.ConvertLoc(
      inst_id,
      [&](DiagnosticLoc /*context_loc*/,
          const Internal::DiagnosticBase<>& /*context_diagnostic_base*/) {});
  return {.filename = diag_loc.filename,
          .line_number = diag_loc.line_number == -1 ? 0 : diag_loc.line_number,
          .column_number =
              diag_loc.column_number == -1 ? 0 : diag_loc.column_number};
}

}  // namespace Carbon::Lower
