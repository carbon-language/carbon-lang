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
#include "toolchain/lower/mangler.h"
#include "toolchain/lower/options.h"
#include "toolchain/lower/specific_coalescer.h"
#include "toolchain/sem_ir/absolute_node_id.h"
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
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/stringify.h"
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
      types_.Set(type_id, BuildType(sem_ir_->types().GetTypeInstId(type_id)));
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
  for (auto inst_id :
       sem_ir().inst_blocks().Get(sem_ir().top_inst_block_id())) {
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

auto FileContext::GetOrCreateFunctionInfo(SemIR::FunctionId function_id,
                                          SemIR::SpecificId specific_id)
    -> std::optional<FunctionInfo>& {
  // If we have already lowered a declaration of this function, just return it.
  auto& result = GetFunctionInfo(function_id, specific_id);
  if (!result) {
    result = BuildFunctionDecl(function_id, specific_id);
  }
  return result;
}

// State machine for building a FunctionTypeInfo from SemIR.
//
// The main difficulty this class encapsulates is that each abstraction level
// has different expectations about how the return is reflected in the parameter
// list.
// - In SemIR, if the function has an initializing return form, it has a
//   corresponding output parameter at the end of the parameter list.
// - In LLVM IR, if the SemIR has an output parameter _and_ that parameter's
//   type has an in-place initializing representation, we emit a corresponding
//   `sret` output parameter (and the function's return type is void). By
//   convention the output parameter goes at the start of the parameter list.
// - In LLVM debug info, the list of parameter types always starts with the
//   return type (which doubles as the type of the return parameter, if there
//   is one).
//
// Furthermore, SemIR is designed to eventually support compound return forms,
// in which case there can be multiple output parameters for different pieces of
// the return form, but it's not yet clear how we will lower such functions.
class FileContext::FunctionTypeInfoBuilder {
 public:
  // Creates a FunctionTypeInfoBuilder that uses the given FileContext, and
  // the given specific of the function.
  FunctionTypeInfoBuilder(FileContext* context, SemIR::SpecificId specific_id)
      : context_(*context), specific_id_(specific_id) {}

  // Retrieves various features of `function`'s type useful for constructing the
  // `llvm::Type` and `llvm::DISubroutineType` for the `llvm::Function`. If any
  // part of the type can't be manifest (eg: incomplete return or parameter
  // types), then the result is as if the type was `void()`. Should only be
  // called once on a given builder.
  auto Build(const SemIR::Function& function) && -> FunctionTypeInfo;

 private:
  // By convention, state transition methods return false to indicate that
  // `Abort` was called. As a convenience, that applies even to methods that
  // never call `Abort`, and to `Abort` itself, so that their callers can easily
  // propagate the failure.

  // Resets the builder to the fallback state `void()`. This puts the builder in
  // a state where Finalize can be called, and no other operation should be
  // called.
  auto Abort() -> bool {
    lowered_param_pattern_ids_.clear();
    param_types_.clear();
    param_di_types_.clear();
    return_type_ = nullptr;
    SetReturnByCopy(SemIR::TypeId::None);
    return false;
  }

  // Handles the function's return form. The argument can be None, indicating
  // that there was no explicitly declared return form.
  //
  // This should be called before HandleParameter. It delegates to exactly one
  // of SetReturnByCopy, SetReturnByReference, SetReturnInPlace, or Abort, and
  // returns false if Abort was called.
  auto HandleReturnForm(SemIR::InstId return_form_inst_id) -> bool;

  // Records that the LLVM function returns by copy, with type `return_type_id`.
  // `return_type_id` can be `None`, which is treated as equivalent to the
  // default return type `()`.
  auto SetReturnByCopy(SemIR::TypeId return_type_id) -> bool {
    CARBON_CHECK(return_type_ == nullptr);
    CARBON_CHECK(param_di_types_.empty());
    auto lowered_return_types = GetLoweredTypes(return_type_id);
    return_type_ = lowered_return_types.llvm_ir_type;
    param_di_types_.push_back(lowered_return_types.llvm_di_type);
    return true;
  }

  // Records that the LLVM function returns by reference, with type
  // `return_type_id`.
  auto SetReturnByReference(SemIR::TypeId /*return_type_id*/) -> bool {
    return_type_ =
        llvm::PointerType::get(context_.llvm_context(), /*AddressSpace=*/0);
    // TODO: replace this with a reference type.
    param_di_types_.push_back(
        context_.context().di_builder().createPointerType(nullptr, 8));
    return true;
  }

  // Records that the LLVM function returns in place, with type
  // `return_type_id`.
  auto SetReturnInPlace(SemIR::TypeId return_type_id) -> bool {
    return_type_ = llvm::Type::getVoidTy(context_.llvm_context());
    sret_type_ = context_.GetType(return_type_id);
    // We don't add to param_di_types_ because that will be handled by the
    // loop over the SemIR parameters.
    return true;
  }

  // Handles the given `Call` parameter, which must be a *ParamPattern inst.
  // This should be called on parameter patterns in the order that they should
  // appear in the LLVM IR parameter list, so in particular it should be called
  // on the `OutParamPattern` (if any) first. It should be called on all `Call`
  // parameters; it will determine which parameters belong in the LLVM IR
  // parameter list.
  //
  // This delegates to exactly one of AddLoweredParam, IgnoreParam, or Abort,
  // and returns false if Abort was called.
  auto HandleParameter(SemIR::InstId param_pattern_id) -> bool;

  // Records that the given parameter pattern is lowered to the given
  // IR and DI types.
  auto AddLoweredParam(SemIR::InstId param_pattern_id, LoweredTypes param_types)
      -> bool {
    lowered_param_pattern_ids_.push_back(param_pattern_id);
    param_types_.push_back(param_types.llvm_ir_type);
    param_di_types_.push_back(param_types.llvm_di_type);
    return true;
  }

  // Records that the given parameter pattern is not lowered to an LLVM
  // parameter.
  auto IgnoreParam(SemIR::InstId param_pattern_id) -> bool {
    unused_param_pattern_ids_.push_back(param_pattern_id);
    return true;
  }

  // Builds and returns a FunctionTypeInfo from the accumulated information.
  auto Finalize() -> FunctionTypeInfo;

  // Returns LLVM IR and DI types for the given SemIR type. This is not a state
  // transition. It mostly delegates to context_.GetTypeAndDIType, but treats
  // TypeId::None as equivalent to the unit type, and uses an untyped pointer as
  // a placeholder DI type if context_ doesn't provide one.
  auto GetLoweredTypes(SemIR::TypeId type_id) -> LoweredTypes;

  FileContext& context_;
  const SemIR::SpecificId specific_id_;

  // The types of the parameters in the LLVM IR function. Each one corresponds
  // to a SemIR `Call` parameter, but some `Call` parameters may be omitted
  // (e.g. if they are stateless) or reordered (e.g. the return parameter, if
  // any, always goes first).
  llvm::SmallVector<llvm::Type*> param_types_;

  // The LLLVM DI representation of the parameter list. As required by LLVM DI
  // convention, this starts with the function's return type, and ends with the
  // DI representations of param_types_ (in the same order). Note that those
  // two ranges may overlap: if the first element of param_types_ represents
  // a return parameter, the first element of param_di_types_ corresponds to it
  // while also representing the return type.
  llvm::SmallVector<llvm::Metadata*> param_di_types_;

  // The SemIR function's `Call` param patterns that correspond to param_types_,
  // in the same order.
  llvm::SmallVector<SemIR::InstId> lowered_param_pattern_ids_;

  // Any `Call` param patterns that aren't present in
  // reordered_param_pattern_ids_.
  llvm::SmallVector<SemIR::InstId> unused_param_pattern_ids_;

  // The `index` member of the SemIR function's return parameter, or -1 if it
  // has no return parameter. Note that even if the SemIR function has a return
  // parameter, the LLVM IR function might not.
  int semir_return_param_index_ = -1;

  // The LLVM function's return type.
  llvm::Type* return_type_ = nullptr;

  // If not null, the LLVM function's first parameter should have a `sret`
  // attribute with this type.
  llvm::Type* sret_type_ = nullptr;
};

auto FileContext::FunctionTypeInfoBuilder::Build(
    const SemIR::Function& function) && -> FunctionTypeInfo {
  // TODO: For the `Run` entry point, remap return type to i32 if it doesn't
  // return a value.

  auto call_param_pattern_ids =
      context_.sem_ir().inst_blocks().Get(function.call_param_patterns_id);
  lowered_param_pattern_ids_.reserve(call_param_pattern_ids.size());
  param_types_.reserve(call_param_pattern_ids.size());
  param_di_types_.reserve(call_param_pattern_ids.size());

  if (!HandleReturnForm(function.return_form_inst_id)) {
    return Finalize();
  }
  if (semir_return_param_index_ >= 0) {
    CARBON_CHECK(semir_return_param_index_ ==
                     static_cast<int>(call_param_pattern_ids.size()) - 1,
                 "Unexpected parameter order");
    // Handle the return parameter first, because it goes first in the LLVM
    // convention. We remove it from call_param_pattern_ids so we don't revisit
    // it in the subsequent loop.
    if (!HandleParameter(call_param_pattern_ids.consume_back())) {
      return Finalize();
    }
  }
  for (auto param_pattern_id : call_param_pattern_ids) {
    if (!HandleParameter(param_pattern_id)) {
      return Finalize();
    }
  }

  return Finalize();
}

auto FileContext::FunctionTypeInfoBuilder::HandleReturnForm(
    SemIR::InstId return_form_inst_id) -> bool {
  if (!return_form_inst_id.has_value()) {
    return SetReturnByCopy(SemIR::TypeId::None);
  }

  auto return_form_const_id = SemIR::GetConstantValueInSpecific(
      context_.sem_ir(), specific_id_, return_form_inst_id);
  auto return_form_inst = context_.sem_ir().insts().Get(
      context_.sem_ir().constant_values().GetInstId(return_form_const_id));
  CARBON_KIND_SWITCH(return_form_inst) {
    case CARBON_KIND(SemIR::InitForm init_form): {
      CARBON_CHECK(
          std::exchange(semir_return_param_index_, init_form.index.index) == -1,
          "TODO: Generalize this to support compound return forms");
      auto return_type_id =
          context_.sem_ir().types().GetTypeIdForTypeConstantId(
              SemIR::GetConstantValueInSpecific(
                  context_.sem_ir(), specific_id_,
                  init_form.type_component_inst_id));
      switch (
          SemIR::InitRepr::ForType(context_.sem_ir(), return_type_id).kind) {
        case SemIR::InitRepr::InPlace: {
          return SetReturnInPlace(return_type_id);
        }
        case SemIR::InitRepr::ByCopy: {
          return SetReturnByCopy(return_type_id);
        }
        case SemIR::InitRepr::None:
          return SetReturnByCopy(SemIR::TypeId::None);
        case SemIR::InitRepr::Dependent:
        case SemIR::InitRepr::Incomplete:
        case SemIR::InitRepr::Abstract:
          return Abort();
      }
    }
    case CARBON_KIND(SemIR::RefForm ref_form): {
      auto return_type_id =
          context_.sem_ir().types().GetTypeIdForTypeConstantId(
              SemIR::GetConstantValueInSpecific(
                  context_.sem_ir(), specific_id_,
                  ref_form.type_component_inst_id));
      return SetReturnByReference(return_type_id);
    }
    default:
      CARBON_FATAL("Unexpected inst kind: {0}", return_form_inst);
  }
}

auto FileContext::FunctionTypeInfoBuilder::HandleParameter(
    SemIR::InstId param_pattern_id) -> bool {
  auto param_pattern = context_.sem_ir().insts().Get(param_pattern_id);
  auto param_type_id = ExtractScrutineeType(
      context_.sem_ir(),
      SemIR::GetTypeOfInstInSpecific(context_.sem_ir(), specific_id_,
                                     param_pattern_id));

  // Returns the appropriate LoweredTypes for reference-like parameters.
  auto ref_lowered_types = [&]() -> LoweredTypes {
    return {.llvm_ir_type = llvm::PointerType::get(context_.llvm_context(),
                                                   /*AddressSpace=*/0),
            // TODO: replace this with a reference type.
            .llvm_di_type = GetLoweredTypes(param_type_id).llvm_di_type};
  };

  CARBON_CHECK(
      !param_type_id.AsConstantId().is_symbolic(),
      "Found symbolic type id after resolution when lowering type {0}.",
      param_pattern.type_id());
  CARBON_KIND_SWITCH(param_pattern) {
    case SemIR::RefParamPattern::Kind:
    case SemIR::VarParamPattern::Kind: {
      return AddLoweredParam(param_pattern_id, ref_lowered_types());
    }
    case SemIR::OutParamPattern::Kind: {
      switch (SemIR::InitRepr::ForType(context_.sem_ir(), param_type_id).kind) {
        case SemIR::InitRepr::InPlace:
          return AddLoweredParam(param_pattern_id, ref_lowered_types());
        case SemIR::InitRepr::ByCopy:
        case SemIR::InitRepr::None:
          return IgnoreParam(param_pattern_id);
        case SemIR::InitRepr::Dependent:
        case SemIR::InitRepr::Incomplete:
        case SemIR::InitRepr::Abstract:
          return Abort();
      }
    }
    case SemIR::ValueParamPattern::Kind: {
      switch (auto value_rep =
                  SemIR::ValueRepr::ForType(context_.sem_ir(), param_type_id);
              value_rep.kind) {
        case SemIR::ValueRepr::Unknown:
          return Abort();
        case SemIR::ValueRepr::Dependent:
          CARBON_FATAL("Lowering function parameter with dependent type: {0}",
                       param_pattern);
        case SemIR::ValueRepr::None:
          return IgnoreParam(param_pattern_id);
        case SemIR::ValueRepr::Copy:
        case SemIR::ValueRepr::Custom:
        case SemIR::ValueRepr::Pointer: {
          if (value_rep.type_id.has_value()) {
            return AddLoweredParam(param_pattern_id,
                                   GetLoweredTypes(value_rep.type_id));
          } else {
            return IgnoreParam(param_pattern_id);
          }
        }
      }
    }
    default:
      CARBON_FATAL("Unexpected inst kind: {0}", param_pattern);
  }
}

auto FileContext::FunctionTypeInfoBuilder::Finalize() -> FunctionTypeInfo {
  CARBON_CHECK(!param_di_types_.empty());
  auto& di_builder = context_.context().di_builder();
  return {.type = llvm::FunctionType::get(return_type_, param_types_,
                                          /*isVarArg=*/false),
          .di_type = di_builder.createSubroutineType(
              di_builder.getOrCreateTypeArray(param_di_types_),
              llvm::DINode::FlagZero),
          .lowered_param_pattern_ids = std::move(lowered_param_pattern_ids_),
          .unused_param_pattern_ids = std::move(unused_param_pattern_ids_),
          .sret_type = sret_type_};
}

auto FileContext::FunctionTypeInfoBuilder::GetLoweredTypes(
    SemIR::TypeId type_id) -> LoweredTypes {
  if (!type_id.has_value()) {
    return {.llvm_ir_type = llvm::Type::getVoidTy(context_.llvm_context()),
            .llvm_di_type = nullptr};
  }
  auto result = context_.GetTypeAndDIType(type_id);
  if (result.llvm_di_type == nullptr) {
    // TODO: figure out what type should go here, or ensure this doesn't
    // happen.
    result.llvm_di_type =
        context_.context().di_builder().createPointerType(nullptr, 8);
  }
  return result;
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

auto FileContext::BuildFunctionDecl(SemIR::FunctionId function_id,
                                    SemIR::SpecificId specific_id)
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

  auto function_type_info =
      FunctionTypeInfoBuilder(this, specific_id).Build(function);

  // TODO: For an imported inline function, consider generating an
  // `available_externally` definition.
  auto linkage = specific_id.has_value() ? llvm::Function::LinkOnceODRLinkage
                                         : llvm::Function::ExternalLinkage;
  if (function_id == sem_ir().global_ctor_id()) {
    // The global constructor name would collide with global constructors for
    // other files in the same package, so use an internal linkage symbol.
    linkage = llvm::Function::InternalLinkage;
  }

  Mangler m(*this);
  std::string mangled_name = m.Mangle(function_id, specific_id);
  if (auto* existing = llvm_module().getFunction(mangled_name)) {
    // We might have already lowered this function while lowering a different
    // file. That's OK.
    // TODO: Check-fail or maybe diagnose if the two LLVM functions are not
    // produced by declarations of the same Carbon function. Name collisions
    // between non-private members of the same library should have been
    // diagnosed by check if detected, but it's not clear that check will always
    // be able to see this problem. In theory, name collisions could also occur
    // due to fingerprint collision.
    return {{.type = function_type_info.type,
             .di_type = function_type_info.di_type,
             .lowered_param_pattern_ids =
                 std::move(function_type_info.lowered_param_pattern_ids),
             .unused_param_pattern_ids =
                 std::move(function_type_info.unused_param_pattern_ids),
             .llvm_function = existing}};
  }

  llvm::Function* llvm_function;
  // If this is a C++ function, tell Clang that we referenced it.
  if (auto clang_decl_id = sem_ir().functions().Get(function_id).clang_decl_id;
      clang_decl_id.has_value()) {
    CARBON_CHECK(!specific_id.has_value(),
                 "Specific functions cannot have C++ definitions");
    llvm_function = HandleReferencedCppFunction(
        sem_ir().clang_decls().Get(clang_decl_id).key.decl->getAsFunction());
  } else {
    // If this is a specific function, we may need to do additional work to emit
    // its definition.
    if (specific_id.has_value()) {
      HandleReferencedSpecificFunction(function_id, specific_id,
                                       function_type_info.type);
    }

    llvm_function = llvm::Function::Create(function_type_info.type, linkage,
                                           mangled_name, llvm_module());

    CARBON_CHECK(llvm_function->getName() == mangled_name,
                 "Mangled name collision: {0}", mangled_name);

    // Set up parameters and the return slot.
    for (auto [inst_id, arg] :
         llvm::zip_equal(function_type_info.lowered_param_pattern_ids,
                         llvm_function->args())) {
      arg.setName(sem_ir().names().GetIRBaseName(
          SemIR::GetPrettyNameFromPatternId(sem_ir(), inst_id)));
    }
    if (function_type_info.sret_type != nullptr) {
      auto& return_arg = *llvm_function->args().begin();
      return_arg.addAttr(llvm::Attribute::getWithStructRetType(
          llvm_context(), function_type_info.sret_type));
    }
  }

  return {{.type = function_type_info.type,
           .di_type = function_type_info.di_type,
           .lowered_param_pattern_ids =
               std::move(function_type_info.lowered_param_pattern_ids),
           .unused_param_pattern_ids =
               std::move(function_type_info.unused_param_pattern_ids),
           .llvm_function = llvm_function}};
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
      definition_context, function_info->llvm_function, *this, specific_id,
      coalescer_.InitializeFingerprintForSpecific(specific_id), subprogram,
      vlog_stream_);

  auto call_param_ids = definition_ir.inst_blocks().GetOrEmpty(
      definition_function.call_params_id);

  // Returns the AnyParam inst with the same index as param_pattern_id
  // (which must be an AnyParamPattern).
  auto param_for_param_pattern =
      [&](SemIR::InstId param_pattern_id) -> SemIR::InstId {
    auto sem_ir_index = sem_ir()
                            .insts()
                            .GetAs<SemIR::AnyParamPattern>(param_pattern_id)
                            .index.index;
    return call_param_ids[sem_ir_index];
  };

  // Add local variables for the parameters.
  for (auto [llvm_index, param_pattern_id] :
       llvm::enumerate(function_info->lowered_param_pattern_ids)) {
    function_lowering.SetLocal(
        param_for_param_pattern(param_pattern_id),
        function_info->llvm_function->getArg(llvm_index));
  }

  // Add local variables for the SemIR parameters that aren't LLVM parameters.
  // These shouldn't actually be used, so they're set to poison values.
  for (auto [llvm_index, param_pattern_id] :
       llvm::enumerate(function_info->unused_param_pattern_ids)) {
    auto param_id = param_for_param_pattern(param_pattern_id);
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
    llvm::BranchInst::Create(entry_block, new_entry_block);
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

// BuildTypeForInst is used to construct types for FileContext::BuildType below.
// Implementations return the LLVM type for the instruction. This first overload
// is the fallback handler for non-type instructions.
template <typename InstT>
  requires(InstT::Kind.is_type() == SemIR::InstIsType::Never)
static auto BuildTypeForInst(FileContext& /*context*/, InstT inst)
    -> FileContext::LoweredTypes {
  CARBON_FATAL("Cannot use inst as type: {0}", inst);
}

template <typename InstT>
  requires(InstT::Kind.is_symbolic_when_type())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> FileContext::LoweredTypes {
  // Treat non-monomorphized symbolic types as opaque.
  return {llvm::StructType::get(context.llvm_context()), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::ArrayType inst)
    -> FileContext::LoweredTypes {
  return {llvm::ArrayType::get(
              context.GetType(context.sem_ir().types().GetTypeIdForTypeInstId(
                  inst.element_type_inst_id)),
              *context.sem_ir().GetArrayBoundValue(inst.bound_id)),
          nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::BoolType /*inst*/)
    -> FileContext::LoweredTypes {
  // TODO: We may want to have different representations for `bool` storage
  // (`i8`) versus for `bool` values (`i1`).
  return {llvm::Type::getInt1Ty(context.llvm_context()), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::ClassType inst)
    -> FileContext::LoweredTypes {
  auto object_repr_id = context.sem_ir()
                            .classes()
                            .Get(inst.class_id)
                            .GetObjectRepr(context.sem_ir(), inst.specific_id);
  return context.GetTypeAndDIType(object_repr_id);
}

template <typename InstT>
  requires(SemIR::Internal::HasInstCategory<SemIR::AnyQualifiedType, InstT>)
static auto BuildTypeForInst(FileContext& context, InstT inst)
    -> FileContext::LoweredTypes {
  return {context.GetType(
              context.sem_ir().types().GetTypeIdForTypeInstId(inst.inner_id)),
          nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::CustomLayoutType inst)
    -> FileContext::LoweredTypes {
  auto layout = context.sem_ir().custom_layouts().Get(inst.layout_id);
  return {llvm::ArrayType::get(llvm::Type::getInt8Ty(context.llvm_context()),
                               layout[SemIR::CustomLayoutId::SizeIndex]),
          nullptr};
}

static auto BuildTypeForInst(FileContext& context,
                             SemIR::ImplWitnessAssociatedConstant inst)
    -> FileContext::LoweredTypes {
  return {context.GetType(inst.type_id), nullptr};
}

static auto BuildTypeForInst(FileContext& /*context*/,
                             SemIR::ErrorInst /*inst*/)
    -> FileContext::LoweredTypes {
  // This is a complete type but uses of it should never be lowered.
  return {nullptr, nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::FloatType inst)
    -> FileContext::LoweredTypes {
  return {llvm::Type::getFloatingPointTy(context.llvm_context(),
                                         inst.float_kind.Semantics()),
          nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::IntType inst)
    -> FileContext::LoweredTypes {
  auto width_inst =
      context.sem_ir().insts().TryGetAs<SemIR::IntValue>(inst.bit_width_id);
  CARBON_CHECK(width_inst, "Can't lower int type with symbolic width");
  auto width = context.sem_ir().ints().Get(width_inst->int_id).getZExtValue();
  return {llvm::IntegerType::get(context.llvm_context(), width),
          context.context().di_builder().createBasicType(
              "int", width,
              inst.int_kind.is_signed() ? llvm::dwarf::DW_ATE_signed
                                        : llvm::dwarf::DW_ATE_unsigned)};
}

static auto BuildTypeForInst(FileContext& context, SemIR::PointerType /*inst*/)
    -> FileContext::LoweredTypes {
  return {llvm::PointerType::get(context.llvm_context(), /*AddressSpace=*/0),
          nullptr};
}

static auto BuildTypeForInst(FileContext& /*context*/,
                             SemIR::PatternType /*inst*/)
    -> FileContext::LoweredTypes {
  CARBON_FATAL("Unexpected pattern type in lowering");
}

static auto BuildTypeForInst(FileContext& context, SemIR::StructType inst)
    -> FileContext::LoweredTypes {
  auto fields = context.sem_ir().struct_type_fields().Get(inst.fields_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  subtypes.reserve(fields.size());
  for (auto field : fields) {
    subtypes.push_back(context.GetType(
        context.sem_ir().types().GetTypeIdForTypeInstId(field.type_inst_id)));
  }
  return {llvm::StructType::get(context.llvm_context(), subtypes), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::TupleType inst)
    -> FileContext::LoweredTypes {
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
  return {llvm::StructType::get(context.llvm_context(), subtypes), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::TypeType /*inst*/)
    -> FileContext::LoweredTypes {
  return {context.GetTypeType(), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::FormType /*inst*/)
    -> FileContext::LoweredTypes {
  return {context.GetFormType(), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::VtableType /*inst*/)
    -> FileContext::LoweredTypes {
  return {llvm::Type::getVoidTy(context.llvm_context()), nullptr};
}

static auto BuildTypeForInst(FileContext& context,
                             SemIR::SpecificFunctionType /*inst*/)
    -> FileContext::LoweredTypes {
  return {llvm::PointerType::get(context.llvm_context(), 0), nullptr};
}

template <typename InstT>
  requires(InstT::Kind.template IsAnyOf<
           SemIR::AssociatedEntityType, SemIR::AutoType, SemIR::BoundMethodType,
           SemIR::CharLiteralType, SemIR::CppOverloadSetType,
           SemIR::CppTemplateNameType, SemIR::FacetType,
           SemIR::FloatLiteralType, SemIR::FunctionType,
           SemIR::FunctionTypeWithSelfType, SemIR::GenericClassType,
           SemIR::GenericInterfaceType, SemIR::GenericNamedConstraintType,
           SemIR::InstType, SemIR::IntLiteralType, SemIR::NamespaceType,
           SemIR::RequireSpecificDefinitionType, SemIR::UnboundElementType,
           SemIR::WhereExpr, SemIR::WitnessType>())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> FileContext::LoweredTypes {
  // Return an empty struct as a placeholder.
  // TODO: Should we model an interface as a witness table, or an associated
  // entity as an index?
  return {llvm::StructType::get(context.llvm_context()), nullptr};
}

auto FileContext::BuildType(SemIR::InstId inst_id) -> LoweredTypes {
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
  auto abs_node_id = GetAbsoluteNodeId(sem_ir_, SemIR::LocId(inst_id)).back();

  if (abs_node_id.check_ir_id() == SemIR::CheckIRId::Cpp) {
    // TODO: Consider asking our cpp_code_generator to map the location to a
    // debug location, in order to use Clang's rules for (eg) macro handling.
    auto loc =
        sem_ir().clang_source_locs().Get(abs_node_id.clang_source_loc_id());
    auto presumed_loc =
        sem_ir().cpp_file()->source_manager().getPresumedLoc(loc);
    return {.filename = presumed_loc.getFilename(),
            .line_number = static_cast<int32_t>(presumed_loc.getLine()),
            .column_number = static_cast<int32_t>(presumed_loc.getColumn())};
  }

  return context().GetLocForDI(abs_node_id);
}

auto FileContext::BuildVtable(const SemIR::Vtable& vtable,
                              SemIR::SpecificId specific_id)
    -> llvm::GlobalVariable* {
  const auto& class_info = sem_ir().classes().Get(vtable.class_id);

  Mangler m(*this);
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
