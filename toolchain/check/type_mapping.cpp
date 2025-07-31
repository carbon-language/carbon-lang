// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/type_mapping.h"

#include <cstddef>
#include <iostream>
#include <optional>

#include "clang/AST/Type.h"
#include "toolchain/base/int.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/literal.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Find the bit width of an integer literal.
// The default bit width is 32. If the literal's bit width is greater than 32,
// the bit width is increased to 64.
static auto FindIntLiteralBitWidth(Context& context, SemIR::InstId arg_id,
                                   bool is_signed) -> IntId {
  auto bit_width_id = IntId::MakeRaw(32);
  auto arg = context.insts().TryGetAs<SemIR::IntValue>(arg_id);
  if (arg) {
    auto arg_val = context.ints().Get(arg->int_id);

    auto width = bit_width_id.has_value()
                     ? context.ints().Get(bit_width_id).getZExtValue()
                     : arg_val.getBitWidth();

    unsigned arg_non_sign_bits = arg_val.getSignificantBits() - 1;
    return (arg_non_sign_bits + is_signed <= width) ? bit_width_id
                                                    : IntId::MakeRaw(64);
  }
  return IntId::MakeRaw(64);
}

// Maps a Carbon record type to a Cpp type.
// Returns `std::nullopt` if the Carbon type is not a ClassType or if the
// Cpp record type has not yet been imported.
static auto TryMapRecordType(Context& context, SemIR::TypeId type_id)
    -> std::optional<clang::QualType> {
  auto class_type =
      context.sem_ir().types().TryGetAs<SemIR::ClassType>(type_id);
  if (!class_type) {
    return std::nullopt;
  }
  const auto& class_info = context.sem_ir().classes().Get(class_type->class_id);
  auto clang_decl_id =
      context.name_scopes().Get(class_info.scope_id).clang_decl_context_id();
  if (!clang_decl_id.has_value()) {
    return std::nullopt;
  }
  clang::Decl* clang_decl =
      context.sem_ir().clang_decls().Get(clang_decl_id).decl;
  auto* record_type_decl = clang::dyn_cast<clang::CXXRecordDecl>(clang_decl);
  return context.ast_context().getRecordType(record_type_decl);
}

// Maps a Carbon builtin type to a Cpp type.
// Returns `std::nullopt` if the type is not supported.
static auto TryMapBuiltInType(Context& context, SemIR::InstId inst_id,
                              SemIR::TypeId type_id)
    -> std::optional<clang::QualType> {
  auto object_repr_id = context.sem_ir().types().GetObjectRepr(type_id);
  if (!object_repr_id.has_value()) {
    return std::nullopt;
  }
  auto type_inst_id = context.sem_ir().types().GetInstId(object_repr_id);
  auto inst = context.insts().Get(type_inst_id);

  clang::QualType mapped_type;
  CARBON_KIND_SWITCH(inst) {
    case SemIR::BoolType::Kind: {
      mapped_type = context.ast_context().BoolTy;
      break;
    }
    case SemIR::IntLiteralType::Kind: {
      auto bit_width = FindIntLiteralBitWidth(context, inst_id, true);
      mapped_type = context.ast_context().getIntTypeForBitwidth(
          bit_width.AsValue(), true);
      break;
    }
    case CARBON_KIND(SemIR::IntType int_type): {
      auto bit_width_inst = context.sem_ir().insts().TryGetAs<SemIR::IntValue>(
          int_type.bit_width_id);
      mapped_type = context.ast_context().getIntTypeForBitwidth(
          bit_width_inst->int_id.AsValue(), int_type.int_kind.is_signed());
      break;
    }
    case SemIR::LegacyFloatType::Kind: {
      mapped_type = context.ast_context().DoubleTy;
      break;
    }
    default: {
      return std::nullopt;
    }
  }
  return mapped_type;
}

// Maps a non-wrapper (no const or pointer) Carbon type to a Cpp type.
// TODO: function that checks if a type is a BuiltinType or a
// RecordType?
static auto MapNonWrapperType(Context& context, SemIR::InstId inst_id,
                              SemIR::TypeId type_id)
    -> std::optional<clang::QualType> {
  auto mapped_type = TryMapBuiltInType(context, inst_id, type_id);
  if (!mapped_type) {
    mapped_type = TryMapRecordType(context, type_id);
  }
  return mapped_type;
}

// TODO: unify this with the C++ to Carbon type mapping function.
auto MapToCppType(Context& context, SemIR::InstId inst_id)
    -> std::optional<clang::QualType> {
  auto type_id = context.insts().Get(inst_id).type_id();
  llvm::SmallVector<SemIR::TypeId> wrapper_types;
  while (true) {
    SemIR::TypeId orig_type_id = type_id;
    if (auto const_type =
            context.sem_ir().types().TryGetAs<SemIR::ConstType>(type_id);
        const_type) {
      type_id =
          context.sem_ir().types().GetTypeIdForTypeInstId(const_type->inner_id);
    } else if (auto pointer_type =
                   context.sem_ir().types().TryGetAs<SemIR::PointerType>(
                       type_id);
               pointer_type) {
      type_id = context.sem_ir().types().GetTypeIdForTypeInstId(
          pointer_type->pointee_id);
    } else {
      break;
    }
    wrapper_types.push_back(orig_type_id);
  }

  std::optional<clang::QualType> mapped_type =
      MapNonWrapperType(context, inst_id, type_id);
  if (!mapped_type) {
    return std::nullopt;
  }

  for (auto wrapper_type_id : llvm::reverse(wrapper_types)) {
    if (auto const_type = context.sem_ir().types().TryGetAs<SemIR::ConstType>(
            wrapper_type_id);
        const_type) {
      mapped_type.value().addConst();
    } else if (auto pointer_type =
                   context.sem_ir().types().TryGetAs<SemIR::PointerType>(
                       wrapper_type_id)) {
      mapped_type = context.ast_context().getPointerType(mapped_type.value());
    } else {
      return std::nullopt;
    }
  }

  return mapped_type;
}

}  // namespace Carbon::Check
