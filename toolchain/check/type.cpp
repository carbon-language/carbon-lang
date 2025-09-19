// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type.h"

#include "toolchain/check/eval.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/facet_type_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto ValidateIntType(Context& context, SemIR::LocId loc_id,
                     SemIR::IntType result) -> bool {
  auto bit_width =
      context.insts().TryGetAs<SemIR::IntValue>(result.bit_width_id);
  if (!bit_width) {
    // Symbolic bit width.
    return true;
  }
  const auto& bit_width_val = context.ints().Get(bit_width->int_id);
  if (bit_width_val.isZero() ||
      (context.types().IsSignedInt(bit_width->type_id) &&
       bit_width_val.isNegative())) {
    CARBON_DIAGNOSTIC(IntWidthNotPositive, Error,
                      "integer type width of {0} is not positive", TypedInt);
    context.emitter().Emit(
        loc_id, IntWidthNotPositive,
        {.type = bit_width->type_id, .value = bit_width_val});
    return false;
  }
  if (bit_width_val.ugt(IntStore::MaxIntWidth)) {
    CARBON_DIAGNOSTIC(IntWidthTooLarge, Error,
                      "integer type width of {0} is greater than the "
                      "maximum supported width of {1}",
                      TypedInt, int);
    context.emitter().Emit(loc_id, IntWidthTooLarge,
                           {.type = bit_width->type_id, .value = bit_width_val},
                           IntStore::MaxIntWidth);
    return false;
  }
  return true;
}

auto ValidateFloatTypeAndSetKind(Context& context, SemIR::LocId loc_id,
                                 SemIR::FloatType& result) -> bool {
  // Get the bit width value.
  auto bit_width_inst =
      context.insts().TryGetAs<SemIR::IntValue>(result.bit_width_id);
  if (!bit_width_inst) {
    // Symbolic bit width. Defer checking until we have a concrete value.
    return true;
  }
  auto bit_width = context.ints().Get(bit_width_inst->int_id);

  // If no kind is specified, infer kind from width.
  if (!result.float_kind.has_value()) {
    switch (bit_width.getLimitedValue()) {
      case 16:
        result.float_kind = SemIR::FloatKind::Binary16;
        break;
      case 32:
        result.float_kind = SemIR::FloatKind::Binary32;
        break;
      case 64:
        result.float_kind = SemIR::FloatKind::Binary64;
        break;
      case 128:
        result.float_kind = SemIR::FloatKind::Binary128;
        break;
      default:
        CARBON_DIAGNOSTIC(CompileTimeFloatBitWidth, Error,
                          "unsupported floating-point bit width {0}", TypedInt);
        context.emitter().Emit(loc_id, CompileTimeFloatBitWidth,
                               TypedInt(bit_width_inst->type_id, bit_width));
        return false;
    }
  }

  if (llvm::APFloat::semanticsSizeInBits(result.float_kind.Semantics()) !=
      bit_width) {
    // This can't currently happen because we don't provide any way to set the
    // float kind other than through the bit width.
    // TODO: Add a float_type.make builtin that takes a float kind, and add a
    // diagnostic here if the size is wrong.
    context.TODO(loc_id, "wrong size for float type");
    return false;
  }

  // TODO: Diagnose if the floating-point type is not supported on this target?

  return true;
}

// Gets or forms a type_id for a type, given the instruction kind and arguments.
template <typename InstT, typename... EachArgT>
static auto GetTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  InstT inst = {SemIR::TypeType::TypeId, each_arg...};
  return context.types().GetTypeIdForTypeConstantId(TryEvalInst(context, inst));
}

// Gets or forms a type_id for a type, given the instruction kind and arguments,
// and completes the type. This should only be used when type completion cannot
// fail.
template <typename InstT, typename... EachArgT>
static auto GetCompleteTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  auto type_id = GetTypeImpl<InstT>(context, each_arg...);
  CompleteTypeOrCheckFail(context, type_id);
  return type_id;
}

auto GetStructType(Context& context, SemIR::StructTypeFieldsId fields_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::StructType>(context, fields_id);
}

auto GetTupleType(Context& context, llvm::ArrayRef<SemIR::InstId> type_inst_ids)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::TupleType>(
      context, context.inst_blocks().AddCanonical(type_inst_ids));
}

auto GetAssociatedEntityType(Context& context, SemIR::InterfaceId interface_id,
                             SemIR::SpecificId interface_specific_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::AssociatedEntityType>(context, interface_id,
                                                  interface_specific_id);
}

auto GetConstType(Context& context, SemIR::TypeInstId inner_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::ConstType>(context, inner_type_id);
}

auto GetQualifiedType(Context& context, SemIR::TypeId type_id,
                      SemIR::TypeQualifiers quals) -> SemIR::TypeId {
  if (quals.HasAnyOf(SemIR::TypeQualifiers::Const)) {
    type_id = GetConstType(context, context.types().GetInstId(type_id));
    quals.Remove(SemIR::TypeQualifiers::Const);
  }
  if (quals.HasAnyOf(SemIR::TypeQualifiers::MaybeUnformed)) {
    type_id = GetTypeImpl<SemIR::MaybeUnformedType>(
        context, context.types().GetInstId(type_id));
    quals.Remove(SemIR::TypeQualifiers::MaybeUnformed);
  }
  if (quals.HasAnyOf(SemIR::TypeQualifiers::Partial)) {
    type_id = GetTypeImpl<SemIR::PartialType>(
        context, context.types().GetInstId(type_id));
    quals.Remove(SemIR::TypeQualifiers::Partial);
  }
  CARBON_CHECK(quals == SemIR::TypeQualifiers::None);
  return type_id;
}

auto GetSingletonType(Context& context, SemIR::TypeInstId singleton_id)
    -> SemIR::TypeId {
  CARBON_CHECK(SemIR::IsSingletonInstId(singleton_id));
  auto type_id = context.types().GetTypeIdForTypeInstId(singleton_id);
  // To keep client code simpler, complete builtin types before returning them.
  CompleteTypeOrCheckFail(context, type_id);
  return type_id;
}

auto GetClassType(Context& context, SemIR::ClassId class_id,
                  SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::ClassType>(context, class_id, specific_id);
}

auto GetCppOverloadSetType(Context& context,
                           SemIR::CppOverloadSetId overload_set_id,
                           SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::CppOverloadSetType>(
      context, overload_set_id, specific_id);
}

auto GetFunctionType(Context& context, SemIR::FunctionId fn_id,
                     SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::FunctionType>(context, fn_id, specific_id);
}

auto GetFunctionTypeWithSelfType(Context& context,
                                 SemIR::TypeInstId interface_function_type_id,
                                 SemIR::InstId self_id) -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::FunctionTypeWithSelfType>(
      context, interface_function_type_id, self_id);
}

auto GetGenericClassType(Context& context, SemIR::ClassId class_id,
                         SemIR::SpecificId enclosing_specific_id)
    -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::GenericClassType>(context, class_id,
                                                      enclosing_specific_id);
}

auto GetGenericInterfaceType(Context& context, SemIR::InterfaceId interface_id,
                             SemIR::SpecificId enclosing_specific_id)
    -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::GenericInterfaceType>(
      context, interface_id, enclosing_specific_id);
}

auto GetInterfaceType(Context& context, SemIR::InterfaceId interface_id,
                      SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::FacetType>(
      context,
      FacetTypeFromInterface(context, interface_id, specific_id).facet_type_id);
}

auto GetFacetType(Context& context, const SemIR::FacetTypeInfo& info)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::FacetType>(context,
                                       context.facet_types().Add(info));
}

auto GetPointerType(Context& context, SemIR::TypeInstId pointee_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::PointerType>(context, pointee_type_id);
}

auto GetPatternType(Context& context, SemIR::TypeId scrutinee_type_id)
    -> SemIR::TypeId {
  CARBON_CHECK(!context.types().Is<SemIR::PatternType>(scrutinee_type_id),
               "Type is already a pattern type");
  if (scrutinee_type_id == SemIR::ErrorInst::TypeId) {
    return SemIR::ErrorInst::TypeId;
  }
  return GetTypeImpl<SemIR::PatternType>(
      context, context.types().GetInstId(scrutinee_type_id));
}

auto GetUnboundElementType(Context& context, SemIR::TypeInstId class_type_id,
                           SemIR::TypeInstId element_type_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::UnboundElementType>(context, class_type_id,
                                                element_type_id);
}

auto GetCanonicalFacetOrTypeValue(Context& context, SemIR::InstId inst_id)
    -> SemIR::InstId {
  auto const_inst_id = context.constant_values().GetConstantInstId(inst_id);

  if (auto access =
          context.insts().TryGetAs<SemIR::FacetAccessType>(const_inst_id)) {
    return access->facet_value_inst_id;
  }

  if (auto access =
          context.insts().TryGetAs<SemIR::SymbolicBindingType>(const_inst_id)) {
    // TODO: Look in ScopeStack with the entity_name_id to find the facet value.
    return access->facet_value_inst_id;
  }

  return const_inst_id;
}

auto GetCanonicalFacetOrTypeValue(Context& context, SemIR::ConstantId const_id)
    -> SemIR::ConstantId {
  return context.constant_values().Get(GetCanonicalFacetOrTypeValue(
      context, context.constant_values().GetInstId(const_id)));
}

}  // namespace Carbon::Check
