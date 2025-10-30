// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/literal.h"

#include "toolchain/check/call.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/name_lookup.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/diagnostics/diagnostic.h"
#include "toolchain/lex/token_info.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

auto MakeIntLiteral(Context& context, Parse::NodeId node_id, IntId int_id)
    -> SemIR::InstId {
  return AddInst<SemIR::IntValue>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::IntLiteralType::TypeInstId),
       .int_id = int_id});
}

auto MakeCharTypeLiteral(Context& context, Parse::NodeId node_id)
    -> SemIR::InstId {
  return LookupNameInCore(context, node_id, "Char");
}

auto MakeIntTypeLiteral(Context& context, Parse::NodeId node_id,
                        SemIR::IntKind int_kind, IntId size_id)
    -> SemIR::InstId {
  auto width_id = MakeIntLiteral(context, node_id, size_id);
  auto fn_inst_id = LookupNameInCore(
      context, node_id, int_kind == SemIR::IntKind::Signed ? "Int" : "UInt");
  return PerformCall(context, node_id, fn_inst_id, {width_id});
}

auto MakeIntType(Context& context, Parse::NodeId node_id,
                 SemIR::IntKind int_kind, IntId size_id) -> SemIR::TypeId {
  auto type_inst_id = MakeIntTypeLiteral(context, node_id, int_kind, size_id);
  return ExprAsType(context, node_id, type_inst_id).type_id;
}

auto MakeFloatTypeLiteral(Context& context, Parse::NodeId node_id,
                          IntId size_id) -> SemIR::InstId {
  auto width_id = MakeIntLiteral(context, node_id, size_id);
  auto fn_inst_id = LookupNameInCore(context, node_id, "Float");
  return PerformCall(context, node_id, fn_inst_id, {width_id});
}

namespace {
// The extracted representation of the type `Core.String`.
struct StringRepr {
  SemIR::TypeId ptr_field_type_id;
  SemIR::TypeId size_field_type_id;
  SemIR::TypeStore::IntTypeInfo size_field_type_info;
};
}  // namespace

// Extracts information about the representation of the `Core.String` type
// necessary for building a string literal.
static auto GetStringLiteralRepr(Context& context, SemIR::LocId loc_id,
                                 SemIR::TypeId type_id)
    -> std::optional<StringRepr> {
  // The object representation should be a struct type.
  auto object_repr_id = context.types().GetObjectRepr(type_id);
  auto struct_repr =
      context.types().TryGetAs<SemIR::StructType>(object_repr_id);
  if (!struct_repr) {
    return std::nullopt;
  }

  // The struct should have two fields.
  auto fields = context.struct_type_fields().Get(struct_repr->fields_id);
  if (fields.size() != 2) {
    return std::nullopt;
  }

  // The first field should be a pointer to 8-bit integers.
  auto ptr_type =
      context.insts().TryGetAs<SemIR::PointerType>(fields[0].type_inst_id);
  if (!ptr_type) {
    return std::nullopt;
  }
  auto pointee_type_id =
      context.types().GetTypeIdForTypeInstId(ptr_type->pointee_id);
  if (!TryToCompleteType(context, pointee_type_id, loc_id)) {
    return std::nullopt;
  }
  auto elem_type_info = context.types().TryGetIntTypeInfo(pointee_type_id);
  if (!elem_type_info || context.ints().Get(elem_type_info->bit_width) != 8) {
    return std::nullopt;
  }

  // The second field should be an integer type.
  auto size_field_type_id =
      context.types().GetTypeIdForTypeInstId(fields[1].type_inst_id);
  auto size_type_info = context.types().TryGetIntTypeInfo(size_field_type_id);
  if (!size_type_info) {
    return std::nullopt;
  }

  return StringRepr{.ptr_field_type_id = context.types().GetTypeIdForTypeInstId(
                        fields[0].type_inst_id),
                    .size_field_type_id = size_field_type_id,
                    .size_field_type_info = *size_type_info};
}

auto MakeStringLiteral(Context& context, Parse::StringLiteralId node_id,
                       StringLiteralValueId value_id) -> SemIR::InstId {
  auto str_type = MakeStringType(context, node_id);
  if (!RequireCompleteType(context, str_type.type_id, node_id, [&] {
        CARBON_DIAGNOSTIC(StringLiteralTypeIncomplete, Error,
                          "type {0} is incomplete", InstIdAsType);
        return context.emitter().Build(node_id, StringLiteralTypeIncomplete,
                                       str_type.inst_id);
      })) {
    return SemIR::ErrorInst::InstId;
  }

  auto repr = GetStringLiteralRepr(context, node_id, str_type.type_id);
  if (!repr) {
    if (str_type.type_id != SemIR::ErrorInst::TypeId) {
      CARBON_DIAGNOSTIC(StringLiteralTypeUnexpected, Error,
                        "unexpected representation for type {0}", InstIdAsType);
      context.emitter().Emit(node_id, StringLiteralTypeUnexpected,
                             str_type.inst_id);
    }
    return SemIR::ErrorInst::InstId;
  }

  // The pointer field is a `StringLiteral` object.
  // TODO: Perhaps `StringLiteral` should instead produce a durable reference,
  // and we should take its address here?
  auto ptr_value_id = AddInst<SemIR::StringLiteral>(
      context, node_id,
      {.type_id = repr->ptr_field_type_id, .string_literal_id = value_id});

  // The size field is an integer literal.
  auto size = context.string_literal_values().Get(value_id).size();
  if (repr->size_field_type_info.bit_width.has_value()) {
    // Check that the size value fits in the size field.
    auto width = context.ints()
                     .Get(repr->size_field_type_info.bit_width)
                     .getLimitedValue();
    if (repr->size_field_type_info.is_signed ? !llvm::isIntN(width, size)
                                             : !llvm::isUIntN(width, size)) {
      CARBON_DIAGNOSTIC(StringLiteralTooLong, Error,
                        "string literal is too long");
      context.emitter().Emit(node_id, StringLiteralTooLong);
      return SemIR::ErrorInst::InstId;
    }
  }
  auto size_value_id =
      AddInst<SemIR::IntValue>(context, node_id,
                               {.type_id = repr->size_field_type_id,
                                .int_id = context.ints().Add(size)});

  // Build the representation struct.
  auto elements_id = context.inst_blocks().Add({ptr_value_id, size_value_id});
  return AddInst<SemIR::StructValue>(
      context, node_id,
      {.type_id = str_type.type_id, .elements_id = elements_id});
}

auto MakeStringTypeLiteral(Context& context, SemIR::LocId loc_id)
    -> SemIR::InstId {
  return LookupNameInCore(context, loc_id, "String");
}

auto MakeStringType(Context& context, SemIR::LocId loc_id) -> TypeExpr {
  auto type_inst_id = MakeStringTypeLiteral(context, loc_id);
  return ExprAsType(context, loc_id, type_inst_id);
}

}  // namespace Carbon::Check
