// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/file_context.h"

#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

FileContext::FileContext(llvm::LLVMContext& llvm_context,
                         llvm::StringRef module_name, const SemIR::File& sem_ir,
                         const SemIR::InstNamer* inst_namer,
                         llvm::raw_ostream* vlog_stream)
    : llvm_context_(&llvm_context),
      llvm_module_(std::make_unique<llvm::Module>(module_name, llvm_context)),
      sem_ir_(&sem_ir),
      inst_namer_(inst_namer),
      vlog_stream_(vlog_stream) {
  CARBON_CHECK(!sem_ir.has_errors())
      << "Generating LLVM IR from invalid SemIR::File is unsupported.";
}

// TODO: Move this to lower.cpp.
auto FileContext::Run() -> std::unique_ptr<llvm::Module> {
  CARBON_CHECK(llvm_module_) << "Run can only be called once.";

  // Lower all types that were required to be complete. Note that this may
  // leave some entries in `types_` null, if those types were mentioned but not
  // used.
  types_.resize(sem_ir_->types().size());
  for (auto type_id : sem_ir_->complete_types()) {
    types_[type_id.index] = BuildType(sem_ir_->types().GetInstId(type_id));
  }

  // Lower function declarations.
  functions_.resize_for_overwrite(sem_ir_->functions().size());
  for (auto i : llvm::seq(sem_ir_->functions().size())) {
    functions_[i] = BuildFunctionDecl(SemIR::FunctionId(i));
  }

  // TODO: Lower global variable declarations.

  // Lower function definitions.
  for (auto i : llvm::seq(sem_ir_->functions().size())) {
    BuildFunctionDefinition(SemIR::FunctionId(i));
  }

  // TODO: Lower global variable initializers.

  return std::move(llvm_module_);
}

auto FileContext::GetGlobal(SemIR::InstId inst_id) -> llvm::Value* {
  auto const_id = sem_ir().constant_values().Get(inst_id);
  if (const_id.is_constant()) {
    inst_id = const_id.inst_id();
  }

  // All builtins are types, with the same empty lowered value.
  if (inst_id.is_builtin()) {
    return GetTypeAsValue();
  }

  auto target = sem_ir().insts().Get(inst_id);
  if (auto function_decl = target.TryAs<SemIR::FunctionDecl>()) {
    return GetFunction(function_decl->function_id);
  }

  if (target.Is<SemIR::AssociatedEntity>() || target.Is<SemIR::FieldDecl>() ||
      target.Is<SemIR::BaseDecl>()) {
    return llvm::ConstantStruct::getAnon(llvm_context(), {});
  }

  if (target.type_id() == SemIR::TypeId::TypeType) {
    return GetTypeAsValue();
  }

  CARBON_FATAL() << "Missing value: " << inst_id << " " << target;
}

auto FileContext::BuildFunctionDecl(SemIR::FunctionId function_id)
    -> llvm::Function* {
  const auto& function = sem_ir().functions().Get(function_id);

  // Don't lower associated functions.
  // TODO: We shouldn't lower any function that has generic parameters.
  if (sem_ir().insts().Is<SemIR::InterfaceDecl>(
          sem_ir().name_scopes().Get(function.enclosing_scope_id).inst_id)) {
    return nullptr;
  }

  // Don't lower builtins.
  if (function.builtin_kind != SemIR::BuiltinFunctionKind::None) {
    return nullptr;
  }

  // Don't lower unused functions.
  if (function.return_slot == SemIR::Function::ReturnSlot::NotComputed) {
    return nullptr;
  }

  const bool has_return_slot = function.has_return_slot();
  auto implicit_param_refs =
      sem_ir().inst_blocks().Get(function.implicit_param_refs_id);
  auto param_refs = sem_ir().inst_blocks().Get(function.param_refs_id);

  SemIR::InitRepr return_rep =
      function.return_type_id.is_valid()
          ? SemIR::GetInitRepr(sem_ir(), function.return_type_id)
          : SemIR::InitRepr{.kind = SemIR::InitRepr::None};
  CARBON_CHECK(return_rep.has_return_slot() == has_return_slot);

  llvm::SmallVector<llvm::Type*> param_types;
  // TODO: Consider either storing `param_inst_ids` somewhere so that we can
  // reuse it from `BuildFunctionDefinition` and when building calls, or factor
  // out a mechanism to compute the mapping between parameters and arguments on
  // demand.
  llvm::SmallVector<SemIR::InstId> param_inst_ids;
  auto max_llvm_params =
      has_return_slot + implicit_param_refs.size() + param_refs.size();
  param_types.reserve(max_llvm_params);
  param_inst_ids.reserve(max_llvm_params);
  if (has_return_slot) {
    param_types.push_back(GetType(function.return_type_id)->getPointerTo());
    param_inst_ids.push_back(function.return_storage_id);
  }
  for (auto param_ref_id :
       llvm::concat<const SemIR::InstId>(implicit_param_refs, param_refs)) {
    auto param_type_id =
        SemIR::Function::GetParamFromParamRefId(sem_ir(), param_ref_id)
            .second.type_id;
    switch (auto value_rep = SemIR::GetValueRepr(sem_ir(), param_type_id);
            value_rep.kind) {
      case SemIR::ValueRepr::Unknown:
        CARBON_FATAL()
            << "Incomplete parameter type lowering function declaration";
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

  // If the initializing representation doesn't produce a value, set the return
  // type to void.
  llvm::Type* return_type = return_rep.kind == SemIR::InitRepr::ByCopy
                                ? GetType(function.return_type_id)
                                : llvm::Type::getVoidTy(llvm_context());

  std::string mangled_name;
  if (SemIR::IsEntryPoint(sem_ir(), function_id)) {
    // TODO: Add an implicit `return 0` if `Run` doesn't return `i32`.
    mangled_name = "main";
  } else if (auto name =
                 sem_ir().names().GetAsStringIfIdentifier(function.name_id)) {
    // TODO: Decide on a name mangling scheme.
    mangled_name = *name;
  } else {
    CARBON_FATAL() << "Unexpected special name for function: "
                   << function.name_id;
  }

  llvm::FunctionType* function_type =
      llvm::FunctionType::get(return_type, param_types, /*isVarArg=*/false);
  auto* llvm_function =
      llvm::Function::Create(function_type, llvm::Function::ExternalLinkage,
                             mangled_name, llvm_module());

  // Set up parameters and the return slot.
  for (auto [inst_id, arg] :
       llvm::zip_equal(param_inst_ids, llvm_function->args())) {
    auto name_id = SemIR::NameId::Invalid;
    if (inst_id == function.return_storage_id) {
      name_id = SemIR::NameId::ReturnSlot;
      arg.addAttr(llvm::Attribute::getWithStructRetType(
          llvm_context(), GetType(function.return_type_id)));
    } else {
      name_id = SemIR::Function::GetParamFromParamRefId(sem_ir(), inst_id)
                    .second.name_id;
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
  FunctionContext function_lowering(*this, llvm_function, vlog_stream_);

  const bool has_return_slot = function.has_return_slot();

  // Add parameters to locals.
  // TODO: This duplicates the mapping between sem_ir instructions and LLVM
  // function parameters that was already computed in BuildFunctionDecl.
  // We should only do that once.
  auto implicit_param_refs =
      sem_ir().inst_blocks().Get(function.implicit_param_refs_id);
  auto param_refs = sem_ir().inst_blocks().Get(function.param_refs_id);
  int param_index = 0;
  if (has_return_slot) {
    function_lowering.SetLocal(function.return_storage_id,
                               llvm_function->getArg(param_index));
    ++param_index;
  }
  for (auto param_ref_id :
       llvm::concat<const SemIR::InstId>(implicit_param_refs, param_refs)) {
    auto [param_id, param] =
        SemIR::Function::GetParamFromParamRefId(sem_ir(), param_ref_id);

    // Get the value of the parameter from the function argument.
    auto param_type_id = param.type_id;
    llvm::Value* param_value = llvm::PoisonValue::get(GetType(param_type_id));
    if (SemIR::GetValueRepr(sem_ir(), param_type_id).kind !=
        SemIR::ValueRepr::None) {
      param_value = llvm_function->getArg(param_index);
      ++param_index;
    }

    // The value of the parameter is the value of the argument.
    function_lowering.SetLocal(param_id, param_value);

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
    // TODO: Should we stop passing compile-time bindings at runtime?
    CARBON_CHECK(bind_name.Is<SemIR::AnyBindName>());
    function_lowering.SetLocal(bind_name_id, param_value);
  }

  // Lower all blocks.
  for (auto block_id : body_block_ids) {
    CARBON_VLOG() << "Lowering " << block_id << "\n";
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

auto FileContext::BuildType(SemIR::InstId inst_id) -> llvm::Type* {
  switch (inst_id.index) {
    case SemIR::BuiltinKind::FloatType.AsInt():
      // TODO: Handle different sizes.
      return llvm::Type::getDoubleTy(*llvm_context_);
    case SemIR::BuiltinKind::IntType.AsInt():
      // TODO: Handle different sizes.
      return llvm::Type::getInt32Ty(*llvm_context_);
    case SemIR::BuiltinKind::BoolType.AsInt():
      // TODO: We may want to have different representations for `bool` storage
      // (`i8`) versus for `bool` values (`i1`).
      return llvm::Type::getInt1Ty(*llvm_context_);
    case SemIR::BuiltinKind::FunctionType.AsInt():
    case SemIR::BuiltinKind::BoundMethodType.AsInt():
    case SemIR::BuiltinKind::NamespaceType.AsInt():
    case SemIR::BuiltinKind::WitnessType.AsInt():
      // Return an empty struct as a placeholder.
      return llvm::StructType::get(*llvm_context_);
    default:
      // Handled below.
      break;
  }

  auto inst = sem_ir_->insts().Get(inst_id);
  switch (inst.kind()) {
    case SemIR::ArrayType::Kind: {
      auto array_type = inst.As<SemIR::ArrayType>();
      return llvm::ArrayType::get(
          GetType(array_type.element_type_id),
          sem_ir_->GetArrayBoundValue(array_type.bound_id));
    }
    case SemIR::AssociatedEntityType::Kind:
      // No runtime operations are provided on an associated entity name, so use
      // an empty representation.
      return llvm::StructType::get(*llvm_context_);
    case SemIR::BindSymbolicName::Kind:
      // Treat non-monomorphized type bindings as opaque.
      return llvm::StructType::get(*llvm_context_);
    case SemIR::ClassType::Kind: {
      auto object_repr_id = sem_ir_->classes()
                                .Get(inst.As<SemIR::ClassType>().class_id)
                                .object_repr_id;
      return GetType(object_repr_id);
    }
    case SemIR::ConstType::Kind:
      return GetType(inst.As<SemIR::ConstType>().inner_id);
    case SemIR::InterfaceType::Kind:
      // Return an empty struct as a placeholder.
      // TODO: Should we model an interface as a witness table?
      return llvm::StructType::get(*llvm_context_);
    case SemIR::IntType::Kind: {
      auto width = sem_ir_->insts().TryGetAs<SemIR::IntLiteral>(
          inst.As<SemIR::IntType>().bit_width_id);
      CARBON_CHECK(width) << "Can't lower int type with symbolic width";
      return llvm::IntegerType::get(
          *llvm_context_, sem_ir_->ints().Get(width->int_id).getZExtValue());
    }
    case SemIR::PointerType::Kind:
      return llvm::PointerType::get(*llvm_context_, /*AddressSpace=*/0);
    case SemIR::StructType::Kind: {
      auto fields =
          sem_ir_->inst_blocks().Get(inst.As<SemIR::StructType>().fields_id);
      llvm::SmallVector<llvm::Type*> subtypes;
      subtypes.reserve(fields.size());
      for (auto field_id : fields) {
        auto field = sem_ir_->insts().GetAs<SemIR::StructTypeField>(field_id);
        subtypes.push_back(GetType(field.field_type_id));
      }
      return llvm::StructType::get(*llvm_context_, subtypes);
    }
    case SemIR::TupleType::Kind: {
      // TODO: Investigate special-casing handling of empty tuples so that they
      // can be collectively replaced with LLVM's void, particularly around
      // function returns. LLVM doesn't allow declaring variables with a void
      // type, so that may require significant special casing.
      auto elements =
          sem_ir_->type_blocks().Get(inst.As<SemIR::TupleType>().elements_id);
      llvm::SmallVector<llvm::Type*> subtypes;
      subtypes.reserve(elements.size());
      for (auto element_id : elements) {
        subtypes.push_back(GetType(element_id));
      }
      return llvm::StructType::get(*llvm_context_, subtypes);
    }
    case SemIR::UnboundElementType::Kind: {
      // Return an empty struct as a placeholder.
      return llvm::StructType::get(*llvm_context_);
    }
    default: {
      CARBON_FATAL() << "Cannot use inst as type: " << inst_id << " " << inst;
    }
  }
}

}  // namespace Carbon::Lower
