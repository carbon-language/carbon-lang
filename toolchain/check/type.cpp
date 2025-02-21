// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type.h"

#include "toolchain/check/eval.h"
#include "toolchain/check/facet_type.h"
#include "toolchain/check/type_completion.h"

namespace Carbon::Check {

// Gets or forms a type_id for a type, given the instruction kind and arguments.
template <typename InstT, typename... EachArgT>
static auto GetTypeImpl(Context& context, EachArgT... each_arg)
    -> SemIR::TypeId {
  // TODO: Remove inst_id parameter from TryEvalInst.
  InstT inst = {SemIR::TypeType::SingletonTypeId, each_arg...};
  return context.types().GetTypeIdForTypeConstantId(
      TryEvalInst(context, SemIR::InstId::None, inst));
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

auto GetTupleType(Context& context, llvm::ArrayRef<SemIR::TypeId> type_ids)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::TupleType>(
      context, context.type_blocks().AddCanonical(type_ids));
}

auto GetAssociatedEntityType(Context& context, SemIR::TypeId interface_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::AssociatedEntityType>(context, interface_type_id);
}

auto GetSingletonType(Context& context, SemIR::InstId singleton_id)
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

auto GetFunctionType(Context& context, SemIR::FunctionId fn_id,
                     SemIR::SpecificId specific_id) -> SemIR::TypeId {
  return GetCompleteTypeImpl<SemIR::FunctionType>(context, fn_id, specific_id);
}

auto GetFunctionTypeWithSelfType(Context& context,
                                 SemIR::InstId interface_function_type_id,
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

auto GetPointerType(Context& context, SemIR::TypeId pointee_type_id)
    -> SemIR::TypeId {
  return GetTypeImpl<SemIR::PointerType>(context, pointee_type_id);
}

auto GetUnboundElementType(Context& context, SemIR::TypeId class_type_id,
                           SemIR::TypeId element_type_id) -> SemIR::TypeId {
  return GetTypeImpl<SemIR::UnboundElementType>(context, class_type_id,
                                                element_type_id);
}

}  // namespace Carbon::Check
