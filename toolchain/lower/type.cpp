// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/type.h"

#include "common/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/file_context.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Lower {

namespace {

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
//
// We also deal with the case where the function signature involves incomplete
// types. This can happen if the function is declared but never defined nor
// called in this file. Declarations of such functions can still need to be
// emitted; currently this happens if they are part of a class's vtable. Such
// uses do not need an exact signature, so we emit them with the LLVM type
// `void()` and set `inexact` on the result to indicate the type is not known.
// LLVM can handle merging inexact and exact signatures, and this matches how
// Clang handles the corresponding situation in C++.
//
// One additional complexity is that we may need to fetch information about the
// same function from multiple different files. For a call to a generic
// function, there may be no single file in which all the relevant types are
// complete, so we will look at both the specific function definition that is
// the resolved callee, as well as the partially-specific function from the call
// site.
//
// In general, we support being given a list of variants of the function, in
// which the first function in the list is the primary declaration and should be
// the most specific function, and the others are used as fallbacks if an
// incomplete type is encountered.
class FunctionTypeInfoBuilder {
 public:
  // Creates a FunctionTypeInfoBuilder that uses the given functions.
  explicit FunctionTypeInfoBuilder(llvm::ArrayRef<FunctionInContext> functions)
      : context_(&functions.front().context->context()), functions_(functions) {
    CARBON_CHECK(!functions_.empty());
  }

  // Retrieves various features of the function's type useful for constructing
  // the `llvm::Type` and `llvm::DISubroutineType` for the `llvm::Function`. If
  // any part of the type can't be manifest (eg: incomplete return or parameter
  // types), then the result is as if the type was `void()`. Should only be
  // called once on a given builder.
  auto Build() && -> FunctionTypeInfo;

 private:
  // By convention, state transition methods return false (without changing the
  // accumulated information about the function) to indicate that we could not
  // manifest the complete function type successfully in this context.

  // Information about how a function is called in SemIR.
  struct SemIRIndexInfo {
    // The number of parameters in the SemIR call signature.
    int num_params;
    // The index of the first return parameter in the SemIR call signature.
    int return_param_index;

    friend auto operator==(const SemIRIndexInfo& lhs, const SemIRIndexInfo& rhs)
        -> bool = default;
  };

  // Get information about the SemIR function signature.
  auto GetSemIRIndexInfo(const FunctionInContext& fn_in_context)
      -> SemIRIndexInfo {
    const auto& sem_ir = fn_in_context.context->sem_ir();
    const auto& function = sem_ir.functions().Get(fn_in_context.function_id);

    int num_params =
        sem_ir.inst_blocks().Get(function.call_param_patterns_id).size();

    int return_param_index = -1;
    if (function.call_param_ranges.return_size() > 0) {
      CARBON_CHECK(function.call_param_ranges.return_size() == 1,
                   "TODO: support multiple return forms");
      return_param_index = function.call_param_ranges.return_begin().index;
    }

    return {.num_params = num_params, .return_param_index = return_param_index};
  }

  // Handles the function's return form.
  //
  // This should be called before `HandleParameter`. It handles the return form
  // by trying each `FunctionInContext` until one succeeds, and returns false if
  // all attempts failed.
  auto HandleReturnForm() -> bool;

  // Tries to handle the return form using the given context. Delegates to
  // exactly one of `SetReturnByCopy`, `SetReturnByReference`, or
  // `SetReturnInPlace`, or returns false if the return type is incomplete.
  auto TryHandleReturnForm(const FunctionInContext& func_ctx) -> bool;

  // Records that the LLVM function returns by copy, with type `return_type_id`.
  // `return_type_id` can be `None`, which is treated as equivalent to the
  // default return type `()`.
  auto SetReturnByCopy(const FunctionInContext& func_ctx,
                       SemIR::TypeId return_type_id) -> bool {
    CARBON_CHECK(return_type_ == nullptr);
    CARBON_CHECK(param_di_types_.empty());
    auto lowered_return_types = GetLoweredTypes(func_ctx, return_type_id);
    return_type_ = lowered_return_types.llvm_ir_type;
    param_di_types_.push_back(lowered_return_types.llvm_di_type);
    return true;
  }

  // Records that the LLVM function returns by reference, with type
  // `return_type_id`.
  auto SetReturnByReference(const FunctionInContext& func_ctx,
                            SemIR::TypeId /*return_type_id*/) -> bool {
    return_type_ = llvm::PointerType::get(func_ctx.context->llvm_context(),
                                          /*AddressSpace=*/0);
    // TODO: replace this with a reference type.
    param_di_types_.push_back(GetPointerDIType(nullptr));
    return true;
  }

  // Records that the LLVM function returns in place, with type
  // `return_type_id`.
  auto SetReturnInPlace(const FunctionInContext& func_ctx,
                        SemIR::TypeId return_type_id) -> bool {
    return_type_ = llvm::Type::getVoidTy(func_ctx.context->llvm_context());
    sret_type_ = func_ctx.context->GetType(return_type_id);
    // We don't add to param_di_types_ because that will be handled by the
    // loop over the SemIR parameters.
    return true;
  }

  // Handles `Call` parameter pattern at the given index. This should be called
  // on parameter patterns in the order that they should appear in the LLVM IR
  // parameter list, so in particular it should be called on the
  // `OutParamPattern` (if any) first. It should be called on all `Call`
  // parameters; it will determine which parameters belong in the LLVM IR
  // parameter list.
  //
  // This tries each `FunctionInContext` until one succeeds, and returns false
  // if all attempts failed.
  auto HandleParameter(SemIR::CallParamIndex index) -> bool;

  // Tries to handle the parameter pattern at the given index using the given
  // context. Delegates to either `AddLoweredParam` or `IgnoreParam`, or returns
  // false if the parameter type is incomplete.
  auto TryHandleParameter(const FunctionInContext& func_ctx,
                          SemIR::CallParamIndex index) -> bool;

  // Records that the parameter pattern at the given index has the given ID, and
  // lowers to the given IR and DI types.
  auto AddLoweredParam(const FunctionInContext& func_ctx,
                       SemIR::CallParamIndex index,
                       SemIR::InstId param_pattern_id, LoweredTypes param_types)
      -> bool {
    lowered_param_indices_.push_back(index);
    param_name_ids_.push_back(SemIR::GetPrettyNameFromPatternId(
        func_ctx.context->sem_ir(), param_pattern_id));
    param_types_.push_back(param_types.llvm_ir_type);
    param_di_types_.push_back(param_types.llvm_di_type);
    return true;
  }

  // Records that the `Call` parameter pattern at the given index is not lowered
  // to an LLVM parameter.
  auto IgnoreParam(SemIR::CallParamIndex index) -> bool {
    unused_param_indices_.push_back(index);
    return true;
  }

  // Builds and returns a FunctionTypeInfo from the accumulated information.
  auto Finalize() -> FunctionTypeInfo;

  // Clears out accumulated state and returns a FunctionTypeInfo with the
  // fallback state `void()`.
  auto Abort() -> FunctionTypeInfo;

  // Returns LLVM IR and DI types for the given SemIR type. This is not a state
  // transition. It mostly delegates to context_.GetTypeAndDIType, but treats
  // TypeId::None as equivalent to the unit type, and uses an untyped pointer as
  // a placeholder DI type if context_ doesn't provide one.
  auto GetLoweredTypes(const FunctionInContext& func_ctx, SemIR::TypeId type_id)
      -> LoweredTypes;

  // Returns a DI type for a pointer to the given pointee. The pointee type may
  // be null.
  auto GetPointerDIType(llvm::DIType* pointee_type, unsigned address_space = 0)
      -> llvm::DIDerivedType* {
    const auto& data_layout = context_->llvm_module().getDataLayout();
    return context_->di_builder().createPointerType(
        pointee_type, data_layout.getPointerSizeInBits(address_space));
  }

  Context* context_;

  llvm::ArrayRef<FunctionInContext> functions_;

  // The number of input `Call` parameter patterns.
  int num_params_ = 0;

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

  // The indices of the `Call` parameters that correspond to `param_types_`, in
  // the same order.
  llvm::SmallVector<SemIR::CallParamIndex> lowered_param_indices_;

  // The names of the `Call` parameters that correspond to `param_types_`, in
  // the same order.
  llvm::SmallVector<SemIR::NameId> param_name_ids_;

  // The indices of any `Call` param patterns that aren't present in
  // lowered_param_indices_.
  llvm::SmallVector<SemIR::CallParamIndex> unused_param_indices_;

  // The LLVM function's return type.
  llvm::Type* return_type_ = nullptr;

  // If not null, the LLVM function's first parameter should have a `sret`
  // attribute with this type.
  llvm::Type* sret_type_ = nullptr;

  // Whether we failed to form an exact description of the function type. This
  // can happen if a parameter or return type is incomplete. In this case, we
  // can still sometimes need to emit a declaration of the function, for example
  // because it appears in a vtable, but we cannot emit a definition or a call.
  bool inexact_ = false;
};

auto FunctionTypeInfoBuilder::Build() && -> FunctionTypeInfo {
  // TODO: For the `Run` entry point, remap return type to i32 if it doesn't
  // return a value.

  // Determine how the parameters are numbered in SemIR, and make sure it's the
  // same for all versions of the function.
  auto semir_info = GetSemIRIndexInfo(functions_.front());
  CARBON_CHECK(llvm::all_of(
      functions_.drop_front(), [&](const FunctionInContext& fn_in_context) {
        return GetSemIRIndexInfo(fn_in_context) == semir_info;
      }));

  num_params_ = semir_info.num_params;
  lowered_param_indices_.reserve(num_params_);
  param_name_ids_.reserve(num_params_);
  param_types_.reserve(num_params_);
  param_di_types_.reserve(num_params_);

  if (!HandleReturnForm()) {
    return Abort();
  }
  int params_end = num_params_;
  if (semir_info.return_param_index >= 0) {
    CARBON_CHECK(semir_info.return_param_index == semir_info.num_params - 1,
                 "Unexpected parameter order");
    params_end = semir_info.return_param_index;
    // Handle the return parameter first, because it goes first in the LLVM
    // convention.
    if (!HandleParameter(
            SemIR::CallParamIndex(semir_info.return_param_index))) {
      return Abort();
    }
  }
  for (int i : llvm::seq(params_end)) {
    if (!HandleParameter(SemIR::CallParamIndex(i))) {
      return Abort();
    }
  }

  return Finalize();
}

auto FunctionTypeInfoBuilder::HandleReturnForm() -> bool {
  for (const auto& func_ctx : functions_) {
    if (TryHandleReturnForm(func_ctx)) {
      return true;
    }
  }
  return false;
}

auto FunctionTypeInfoBuilder::TryHandleReturnForm(
    const FunctionInContext& func_ctx) -> bool {
  const auto& function =
      func_ctx.context->sem_ir().functions().Get(func_ctx.function_id);
  auto return_form_inst_id = function.return_form_inst_id;
  if (!return_form_inst_id.has_value()) {
    return SetReturnByCopy(func_ctx, SemIR::TypeId::None);
  }

  auto return_form_const_id = SemIR::GetConstantValueInSpecific(
      func_ctx.context->sem_ir(), func_ctx.specific_id, return_form_inst_id);
  auto return_form_inst = func_ctx.context->sem_ir().insts().Get(
      func_ctx.context->sem_ir().constant_values().GetInstId(
          return_form_const_id));
  CARBON_KIND_SWITCH(return_form_inst) {
    case CARBON_KIND(SemIR::InitForm init_form): {
      auto return_type_id =
          func_ctx.context->sem_ir().types().GetTypeIdForTypeConstantId(
              SemIR::GetConstantValueInSpecific(
                  func_ctx.context->sem_ir(), func_ctx.specific_id,
                  init_form.type_component_inst_id));
      switch (
          SemIR::InitRepr::ForType(func_ctx.context->sem_ir(), return_type_id)
              .kind) {
        case SemIR::InitRepr::InPlace:
          return SetReturnInPlace(func_ctx, return_type_id);
        case SemIR::InitRepr::ByCopy:
          return SetReturnByCopy(func_ctx, return_type_id);
        case SemIR::InitRepr::None:
          return SetReturnByCopy(func_ctx, SemIR::TypeId::None);
        case SemIR::InitRepr::Dependent:
          CARBON_FATAL("Lowering function return with dependent type: {0}",
                       return_form_inst);
        case SemIR::InitRepr::Incomplete:
        case SemIR::InitRepr::Abstract:
          return false;
      }
    }
    case CARBON_KIND(SemIR::RefForm ref_form): {
      auto return_type_id =
          func_ctx.context->sem_ir().types().GetTypeIdForTypeConstantId(
              SemIR::GetConstantValueInSpecific(
                  func_ctx.context->sem_ir(), func_ctx.specific_id,
                  ref_form.type_component_inst_id));
      return SetReturnByReference(func_ctx, return_type_id);
    }
    case CARBON_KIND(SemIR::ValueForm val_form): {
      auto return_type_id =
          func_ctx.context->sem_ir().types().GetTypeIdForTypeConstantId(
              SemIR::GetConstantValueInSpecific(
                  func_ctx.context->sem_ir(), func_ctx.specific_id,
                  val_form.type_component_inst_id));
      switch (
          SemIR::ValueRepr::ForType(func_ctx.context->sem_ir(), return_type_id)
              .kind) {
        case SemIR::ValueRepr::Unknown:
          return false;
        case SemIR::ValueRepr::Dependent:
          CARBON_FATAL("Lowering function return with dependent type: {0}",
                       return_form_inst);
        case SemIR::ValueRepr::None:
          return SetReturnByCopy(func_ctx, SemIR::TypeId::None);
        case SemIR::ValueRepr::Copy:
          return SetReturnByCopy(func_ctx, return_type_id);
        case SemIR::ValueRepr::Pointer:
        case SemIR::ValueRepr::Custom:
          return SetReturnByReference(func_ctx, return_type_id);
      }
    }
    default:
      CARBON_FATAL("Unexpected inst kind: {0}", return_form_inst);
  }
}

auto FunctionTypeInfoBuilder::HandleParameter(SemIR::CallParamIndex index)
    -> bool {
  for (const auto& func_ctx : functions_) {
    if (TryHandleParameter(func_ctx, index)) {
      return true;
    }
  }
  return false;
}

auto FunctionTypeInfoBuilder::TryHandleParameter(
    const FunctionInContext& func_ctx, SemIR::CallParamIndex index) -> bool {
  const auto& sem_ir = func_ctx.context->sem_ir();
  auto param_pattern_id =
      sem_ir.inst_blocks().Get(sem_ir.functions()
                                   .Get(func_ctx.function_id)
                                   .call_param_patterns_id)[index.index];
  auto param_pattern = sem_ir.insts().Get(param_pattern_id);
  auto param_type_id = ExtractScrutineeType(
      sem_ir, SemIR::GetTypeOfInstInSpecific(sem_ir, func_ctx.specific_id,
                                             param_pattern_id));

  // Returns the appropriate LoweredTypes for reference-like parameters.
  auto ref_lowered_types = [&]() -> LoweredTypes {
    return {
        .llvm_ir_type = llvm::PointerType::get(func_ctx.context->llvm_context(),
                                               /*AddressSpace=*/0),
        // TODO: replace this with a reference type.
        .llvm_di_type = GetLoweredTypes(func_ctx, param_type_id).llvm_di_type};
  };

  CARBON_CHECK(
      !param_type_id.AsConstantId().is_symbolic(),
      "Found symbolic type id after resolution when lowering type {0}.",
      param_pattern.type_id());

  auto param_kind = param_pattern.kind();

  // Treat a form parameter pattern like the kind of param pattern that
  // corresponds to its form.
  if (auto form_param_pattern =
          param_pattern.TryAs<SemIR::FormParamPattern>()) {
    auto form_kind = sem_ir.insts().Get(form_param_pattern->form_id).kind();
    switch (form_kind) {
      case SemIR::InitForm::Kind:
        param_kind = SemIR::VarParamPattern::Kind;
        break;
      case SemIR::RefForm::Kind:
        param_kind = SemIR::RefParamPattern::Kind;
        break;
      case SemIR::ValueForm::Kind:
        param_kind = SemIR::ValueParamPattern::Kind;
        break;
      default:
        CARBON_FATAL("Unexpected kind {0} for form inst", form_kind);
    }
  }

  switch (param_kind) {
    case SemIR::RefParamPattern::Kind:
    case SemIR::VarParamPattern::Kind: {
      return AddLoweredParam(func_ctx, index, param_pattern_id,
                             ref_lowered_types());
    }
    case SemIR::OutParamPattern::Kind: {
      switch (SemIR::InitRepr::ForType(sem_ir, param_type_id).kind) {
        case SemIR::InitRepr::InPlace:
          return AddLoweredParam(func_ctx, index, param_pattern_id,
                                 ref_lowered_types());
        case SemIR::InitRepr::ByCopy:
        case SemIR::InitRepr::None:
          return IgnoreParam(index);
        case SemIR::InitRepr::Dependent:
          CARBON_FATAL("Lowering function parameter with dependent type: {0}",
                       param_pattern);
        case SemIR::InitRepr::Incomplete:
        case SemIR::InitRepr::Abstract:
          return false;
      }
    }
    case SemIR::ValueParamPattern::Kind: {
      switch (auto value_rep = SemIR::ValueRepr::ForType(sem_ir, param_type_id);
              value_rep.kind) {
        case SemIR::ValueRepr::Unknown:
          return false;
        case SemIR::ValueRepr::Dependent:
          CARBON_FATAL("Lowering function parameter with dependent type: {0}",
                       param_pattern);
        case SemIR::ValueRepr::None:
          return IgnoreParam(index);
        case SemIR::ValueRepr::Copy:
        case SemIR::ValueRepr::Custom:
        case SemIR::ValueRepr::Pointer: {
          if (value_rep.type_id.has_value()) {
            return AddLoweredParam(
                func_ctx, index, param_pattern_id,
                GetLoweredTypes(func_ctx, value_rep.type_id));
          } else {
            return IgnoreParam(index);
          }
        }
      }
    }
    default:
      CARBON_FATAL("Unexpected inst kind: {0}", param_pattern);
  }
}

auto FunctionTypeInfoBuilder::Finalize() -> FunctionTypeInfo {
  CARBON_CHECK(lowered_param_indices_.size() + unused_param_indices_.size() ==
               static_cast<size_t>(num_params_));
  CARBON_CHECK(!param_di_types_.empty());
  auto& di_builder = context_->di_builder();
  return {.type = llvm::FunctionType::get(return_type_, param_types_,
                                          /*isVarArg=*/false),
          .di_type = di_builder.createSubroutineType(
              di_builder.getOrCreateTypeArray(param_di_types_),
              llvm::DINode::FlagZero),
          .lowered_param_indices = std::move(lowered_param_indices_),
          .unused_param_indices = std::move(unused_param_indices_),
          .param_name_ids = std::move(param_name_ids_),
          .sret_type = sret_type_,
          .inexact = inexact_};
}

auto FunctionTypeInfoBuilder::Abort() -> FunctionTypeInfo {
  num_params_ = 0;
  lowered_param_indices_.clear();
  unused_param_indices_.clear();
  param_name_ids_.clear();
  param_types_.clear();
  param_di_types_.clear();
  return_type_ = llvm::Type::getVoidTy(context_->llvm_context());
  param_di_types_.push_back(nullptr);
  inexact_ = true;
  return Finalize();
}

auto FunctionTypeInfoBuilder::GetLoweredTypes(const FunctionInContext& func_ctx,
                                              SemIR::TypeId type_id)
    -> LoweredTypes {
  if (!type_id.has_value()) {
    return {
        .llvm_ir_type = llvm::Type::getVoidTy(func_ctx.context->llvm_context()),
        .llvm_di_type = nullptr};
  }
  auto result = func_ctx.context->GetTypeAndDIType(type_id);
  if (result.llvm_di_type == nullptr) {
    // TODO: figure out what type should go here, or ensure this doesn't
    // happen.
    result.llvm_di_type = GetPointerDIType(nullptr);
  }
  return result;
}

}  // namespace

auto BuildFunctionTypeInfo(llvm::ArrayRef<FunctionInContext> functions)
    -> FunctionTypeInfo {
  return FunctionTypeInfoBuilder(functions).Build();
}

// Given an LLVM type, build a corresponding type with `padding_bytes` bytes of
// explicit tail padding.
static auto BuildTailPaddedType(llvm::Type* subtype, int64_t padding_bytes)
    -> llvm::Type* {
  if (padding_bytes == 0) {
    return subtype;
  }
  // Build the type `<{subtype, [i8 x padding_bytes]}>`.
  llvm::Type* type_with_padding[2] = {
      subtype,
      llvm::ArrayType::get(llvm::Type::getInt8Ty(subtype->getContext()),
                           padding_bytes)};
  return llvm::StructType::get(subtype->getContext(), type_with_padding,
                               /*isPacked=*/true);
}

// BuildTypeForInst is used to construct types for FileContext::BuildType below.
// Implementations return the LLVM type for the instruction. This first overload
// is the fallback handler for non-type instructions.
template <typename InstT>
  requires(InstT::Kind.is_type() == SemIR::InstIsType::Never)
static auto BuildTypeForInst(FileContext& /*context*/, InstT inst)
    -> LoweredTypes {
  CARBON_FATAL("Cannot use inst as type: {0}", inst);
}

template <typename InstT>
  requires(InstT::Kind.is_symbolic_when_type())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> LoweredTypes {
  // Treat non-monomorphized symbolic types as opaque.
  return {llvm::StructType::get(context.llvm_context()), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::ArrayType inst)
    -> LoweredTypes {
  auto elem_type_id = context.sem_ir().types().GetTypeIdForTypeInstId(
      inst.element_type_inst_id);
  auto stride = context.sem_ir()
                    .types()
                    .GetCompleteTypeInfo(elem_type_id)
                    .object_layout.ArrayStride();

  auto* elem_type = context.GetType(elem_type_id);
  auto elem_size = SemIR::ObjectSize::Bytes(
      context.llvm_module().getDataLayout().getTypeAllocSize(elem_type));

  if (elem_size != stride) {
    CARBON_CHECK(elem_size < stride, "Array element type too large");
    elem_type = BuildTailPaddedType(context.GetType(elem_type_id),
                                    stride.bytes() - elem_size.bytes());
  }

  return {llvm::ArrayType::get(
              elem_type, *context.sem_ir().GetZExtIntValue(inst.bound_id)),
          nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::BoolType /*inst*/)
    -> LoweredTypes {
  // TODO: We may want to have different representations for `bool` storage
  // (`i8`) versus for `bool` values (`i1`).
  return {llvm::Type::getInt1Ty(context.llvm_context()), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::ClassType inst)
    -> LoweredTypes {
  auto object_repr_id = context.sem_ir()
                            .classes()
                            .Get(inst.class_id)
                            .GetObjectRepr(context.sem_ir(), inst.specific_id);
  return context.GetTypeAndDIType(object_repr_id);
}

template <typename InstT>
  requires(SemIR::Internal::HasInstCategory<SemIR::AnyQualifiedType, InstT>)
static auto BuildTypeForInst(FileContext& context, InstT inst) -> LoweredTypes {
  return {context.GetType(
              context.sem_ir().types().GetTypeIdForTypeInstId(inst.inner_id)),
          nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::CustomLayoutType inst)
    -> LoweredTypes {
  auto layout = context.sem_ir().custom_layouts().Get(inst.layout_id);
  return {
      llvm::ArrayType::get(llvm::Type::getInt8Ty(context.llvm_context()),
                           layout[SemIR::CustomLayoutId::SizeIndex].bytes()),
      nullptr};
}

static auto BuildTypeForInst(FileContext& context,
                             SemIR::ImplWitnessAssociatedConstant inst)
    -> LoweredTypes {
  return {context.GetType(inst.type_id), nullptr};
}

static auto BuildTypeForInst(FileContext& /*context*/,
                             SemIR::ErrorInst /*inst*/) -> LoweredTypes {
  // This is a complete type but uses of it should never be lowered.
  return {nullptr, nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::FloatType inst)
    -> LoweredTypes {
  return {llvm::Type::getFloatingPointTy(context.llvm_context(),
                                         inst.float_kind.Semantics()),
          nullptr};
}

static auto BuildTypeForInst(FileContext& /*context*/,
                             SemIR::ImplWitnessAccess /*inst*/)
    -> LoweredTypes {
  CARBON_FATAL("Unexpected ImplWitnessAccess in lowering");
}

static auto BuildTypeForInst(FileContext& context, SemIR::IntType inst)
    -> LoweredTypes {
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
    -> LoweredTypes {
  return {llvm::PointerType::get(context.llvm_context(), /*AddressSpace=*/0),
          nullptr};
}

static auto BuildTypeForInst(FileContext& /*context*/,
                             SemIR::PatternType /*inst*/) -> LoweredTypes {
  CARBON_FATAL("Unexpected pattern type in lowering");
}

// Builds an LLVM packed struct type whose layout matches the Carbon layout for
// an aggregate with the given field types and field layouts.
static auto BuildPackedStructType(FileContext& context,
                                  llvm::MutableArrayRef<llvm::Type*> subtypes,
                                  llvm::ArrayRef<SemIR::ObjectLayout> layouts)
    -> llvm::StructType* {
  const auto& data_layout = context.llvm_module().getDataLayout();
  auto struct_layout = SemIR::ObjectLayout::Empty();
  auto size_so_far = SemIR::ObjectSize::Zero();

  llvm::Type** previous_type = nullptr;
  for (auto [type, layout] : llvm::zip_equal(subtypes, layouts)) {
    auto offset = struct_layout.FieldOffset(layout);
    // If this field has padding before it, represent that padding explicitly as
    // part of the previous field. This allows us to always use GEP indexes that
    // match the field indexes.
    if (offset != size_so_far) {
      CARBON_CHECK(previous_type, "Padding before first field?");
      CARBON_CHECK(offset > size_so_far, "Extraneous padding after field {0}",
                   **previous_type);
      int64_t padding_bytes = offset.bytes() - struct_layout.size.bytes();
      *previous_type = BuildTailPaddedType(*previous_type, padding_bytes);
      size_so_far += SemIR::ObjectSize::Bytes(padding_bytes);
      CARBON_CHECK(offset == size_so_far, "Field at non-byte offset");
    }

    size_so_far += SemIR::ObjectSize::Bytes(data_layout.getTypeAllocSize(type));
    struct_layout.AppendField(layout);
    previous_type = &type;
  }
  return llvm::StructType::get(context.llvm_context(), subtypes,
                               /*isPacked=*/true);
}

// Returns whether the given LLVM layout matches the expected Carbon layout for
// an aggregate with the given field layouts.
static auto StructLayoutMatches(llvm::ArrayRef<SemIR::ObjectLayout> layouts,
                                const llvm::StructLayout& llvm_layout) -> bool {
  auto struct_layout = SemIR::ObjectLayout::Empty();

  // Check each field is at the right offset.
  for (auto [i, layout] : llvm::enumerate(layouts)) {
    if (static_cast<int64_t>(llvm_layout.getElementOffsetInBits(i)) !=
        struct_layout.FieldOffset(layout).bits()) {
      return false;
    }
    struct_layout.AppendField(layout);
  }

  // Treat the LLVM layout as being acceptable if it's the right byte size and
  // does not require more alignment than the Carbon type. We could ignore the
  // alignment, but an overaligned LLVM type will prevent the type from being
  // used in non-packed structs in more situations.
  return static_cast<int64_t>(llvm_layout.getSizeInBytes()) ==
             struct_layout.size.bytes() &&
         llvm_layout.getAlignment() <=
             llvm::Align(struct_layout.alignment.bytes());
}

// Builds an LLVM struct type whose layout matches the Carbon layout for an
// aggregate with the given field types and field layouts.
static auto BuildStructType(FileContext& context,
                            llvm::MutableArrayRef<llvm::Type*> subtypes,
                            llvm::ArrayRef<SemIR::ObjectLayout> layouts)
    -> LoweredTypes {
  // Opportunistically try building an llvm StructType from the subtypes. If it
  // has the right layout, we're done. We prefer to use a non-packed struct type
  // where possible to produce a smaller LLVM IR representation for the type and
  // for constant values of the type, and to improve the readability of the IR.
  auto* struct_type = llvm::StructType::get(context.llvm_context(), subtypes);
  if (!StructLayoutMatches(
          layouts, *context.llvm_module().getDataLayout().getStructLayout(
                       struct_type))) {
    struct_type = BuildPackedStructType(context, subtypes, layouts);
  }
  return {struct_type, nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::StructType inst)
    -> LoweredTypes {
  auto fields = context.sem_ir().struct_type_fields().Get(inst.fields_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  llvm::SmallVector<SemIR::ObjectLayout> layouts;
  subtypes.reserve(fields.size());
  layouts.reserve(fields.size());
  for (auto field : fields) {
    auto type_id =
        context.sem_ir().types().GetTypeIdForTypeInstId(field.type_inst_id);
    subtypes.push_back(context.GetType(type_id));
    layouts.push_back(
        context.sem_ir().types().GetCompleteTypeInfo(type_id).object_layout);
  }
  return BuildStructType(context, subtypes, layouts);
}

static auto BuildTypeForInst(FileContext& context, SemIR::TupleType inst)
    -> LoweredTypes {
  // TODO: Investigate special-casing handling of empty tuples so that they
  // can be collectively replaced with LLVM's void, particularly around
  // function returns. LLVM doesn't allow declaring variables with a void
  // type, so that may require significant special casing.
  auto elements = context.sem_ir().inst_blocks().Get(inst.type_elements_id);
  llvm::SmallVector<llvm::Type*> subtypes;
  llvm::SmallVector<SemIR::ObjectLayout> layouts;
  subtypes.reserve(elements.size());
  layouts.reserve(elements.size());
  for (auto type_id : context.sem_ir().types().GetBlockAsTypeIds(elements)) {
    subtypes.push_back(context.GetType(type_id));
    layouts.push_back(
        context.sem_ir().types().GetCompleteTypeInfo(type_id).object_layout);
  }
  return BuildStructType(context, subtypes, layouts);
}

static auto BuildTypeForInst(FileContext& context, SemIR::TypeType /*inst*/)
    -> LoweredTypes {
  return {context.GetTypeType(), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::FormType /*inst*/)
    -> LoweredTypes {
  return {context.GetFormType(), nullptr};
}

static auto BuildTypeForInst(FileContext& context, SemIR::VtableType /*inst*/)
    -> LoweredTypes {
  return {llvm::Type::getVoidTy(context.llvm_context()), nullptr};
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
           SemIR::RequireSpecificDefinitionType, SemIR::SpecificFunctionType,
           SemIR::UnboundElementType, SemIR::WhereExpr, SemIR::WitnessType>())
static auto BuildTypeForInst(FileContext& context, InstT /*inst*/)
    -> LoweredTypes {
  // Return an empty struct as a placeholder.
  // TODO: Should we model an interface as a witness table, or an associated
  // entity as an index?
  return {llvm::StructType::get(context.llvm_context()), nullptr};
}

auto BuildType(FileContext& context, SemIR::InstId inst_id) -> LoweredTypes {
  // Use overload resolution to select the implementation, producing compile
  // errors when BuildTypeForInst isn't defined for a given instruction.
  LoweredTypes result;
  CARBON_KIND_SWITCH(context.sem_ir().insts().Get(inst_id)) {
#define CARBON_SEM_IR_INST_KIND(Name)         \
  case CARBON_KIND(SemIR::Name inst): {       \
    result = BuildTypeForInst(context, inst); \
    break;                                    \
  }
#include "toolchain/sem_ir/inst_kind.def"
  }

  // In debug builds, check that the type we built has the expected size.
  CARBON_DCHECK([&] {
    if (!result.llvm_ir_type) {
      return true;
    }
    const auto& layout = context.llvm_module().getDataLayout();
    auto expected_layout =
        context.sem_ir()
            .types()
            .GetCompleteTypeInfo(
                context.sem_ir().types().GetTypeIdForTypeInstId(inst_id))
            .object_layout;
    CARBON_CHECK(expected_layout.has_value());
    auto size =
        SemIR::ObjectSize::Bits(layout.getTypeSizeInBits(result.llvm_ir_type));
    // Round up to byte granularity for this check, since LLVM doesn't support
    // non-byte-sized packed structs.
    CARBON_CHECK(
        size.bytes() == expected_layout.size.bytes(),
        "Lowered type {0} for {1} has unexpected size {2}, expected {3}",
        *result.llvm_ir_type, context.sem_ir().insts().Get(inst_id), size,
        expected_layout.size);
    return true;
  }());

  return result;
}

}  // namespace Carbon::Lower
