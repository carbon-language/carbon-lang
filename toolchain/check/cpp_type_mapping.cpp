// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp_type_mapping.h"

#include <cstddef>
#include <iostream>
#include <optional>

#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"
#include "toolchain/base/int.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/literal.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Find the bit width of an integer literal.
// The default bit width is 32. If the literal's bit width is greater than 32,
// the bit width is increased to 64.
static auto FindIntLiteralBitWidth(Context& context, SemIR::InstId arg_id)
    -> IntId {
  auto arg_const_id = context.constant_values().Get(arg_id);
  if (!arg_const_id.is_constant() ||
      arg_const_id == SemIR::ErrorInst::ConstantId ||
      arg_const_id.is_symbolic()) {
    // TODO: Add tests for these cases.
    return IntId::None;
  }
  auto arg = context.insts().GetAs<SemIR::IntValue>(
      context.constant_values().GetInstId(arg_const_id));
  unsigned arg_non_sign_bits =
      context.ints().Get(arg.int_id).getSignificantBits() - 1;

  // TODO: What if the literal is larger than 64 bits? Currently an error is
  // reported that the int value is too large for type `i64`. Maybe try to fit
  // in i128/i256? Try unsigned?
  return (arg_non_sign_bits <= 32) ? IntId::MakeRaw(32) : IntId::MakeRaw(64);
}

// Maps a Carbon builtin type to a Cpp type. Returns an empty QualType if the
// type is not supported.
// TODO: Have both Carbon -> C++ and C++ -> Carbon mappings in a single place
// to keep them in sync.
static auto TryMapBuiltinType(Context& context, SemIR::InstId inst_id,
                              SemIR::TypeId type_id) -> clang::QualType {
  // TODO: Object representation should not be used here. Instead, use
  // NumericTypeLiteralInfo to decompose a ClassType into a uN / iN / fN type.
  auto object_repr_id = context.sem_ir().types().GetObjectRepr(type_id);
  if (!object_repr_id.has_value()) {
    return clang::QualType();
  }
  auto type_inst =
      context.insts().Get(context.sem_ir().types().GetInstId(object_repr_id));

  CARBON_KIND_SWITCH(type_inst) {
    case SemIR::BoolType::Kind: {
      return context.ast_context().BoolTy;
    }
    case Carbon::SemIR::CharLiteralType::Kind: {
      return context.ast_context().CharTy;
    }
    case SemIR::IntLiteralType::Kind: {
      IntId bit_width_id = FindIntLiteralBitWidth(context, inst_id);
      if (bit_width_id == IntId::None) {
        return clang::QualType();
      }
      return context.ast_context().getIntTypeForBitwidth(bit_width_id.AsValue(),
                                                         true);
    }
    case CARBON_KIND(SemIR::IntType int_type): {
      auto bit_width_inst = context.sem_ir().insts().TryGetAs<SemIR::IntValue>(
          int_type.bit_width_id);
      return context.ast_context().getIntTypeForBitwidth(
          bit_width_inst->int_id.AsValue(), int_type.int_kind.is_signed());
    }
    // TODO: What if the value doesn't fit to f64?
    case SemIR::FloatLiteralType::Kind: {
      return context.ast_context().DoubleTy;
    }
    case CARBON_KIND(SemIR::FloatType float_type): {
      auto bit_width_inst = context.sem_ir().insts().TryGetAs<SemIR::IntValue>(
          float_type.bit_width_id);
      return context.ast_context().getRealTypeForBitwidth(
          bit_width_inst->int_id.AsValue(), clang::FloatModeKind::NoFloat);
    }
    default: {
      return clang::QualType();
    }
  }
}

// Maps a Carbon class type to a C++ type. Returns a null `QualType` if the
// Carbon type is not a `ClassType` or was not imported from C++.
// TODO: If the class type wasn't imported from C++, create a corresponding C++
// class type.
static auto TryMapClassType(Context& context, SemIR::TypeId type_id)
    -> clang::QualType {
  auto class_type =
      context.sem_ir().types().TryGetAs<SemIR::ClassType>(type_id);
  if (!class_type) {
    return clang::QualType();
  }

  auto clang_decl_id =
      context.name_scopes()
          .Get(context.sem_ir().classes().Get(class_type->class_id).scope_id)
          .clang_decl_context_id();
  if (!clang_decl_id.has_value()) {
    return clang::QualType();
  }
  clang::Decl* clang_decl =
      context.sem_ir().clang_decls().Get(clang_decl_id).decl;
  auto* tag_type_decl = clang::cast<clang::TagDecl>(clang_decl);
  return context.ast_context().getCanonicalTagType(tag_type_decl);
}

// Maps a non-wrapper (no const or pointer) Carbon type to a C++ type.
static auto MapNonWrapperType(Context& context, SemIR::InstId inst_id,
                              SemIR::TypeId type_id) -> clang::QualType {
  // It's important to check for a class type first, because an enum imported
  // from C++ is both a Carbon class type and has an object representation that
  // is a builtin type, and we want to map back to the enum.
  // TODO: The order won't matter once TryMapBuiltinType stops looking through
  // adapters.
  clang::QualType mapped_type = TryMapClassType(context, type_id);
  if (mapped_type.isNull()) {
    mapped_type = TryMapBuiltinType(context, inst_id, type_id);
  }
  return mapped_type;
}

// TODO: unify this with the C++ to Carbon type mapping function.
auto MapToCppType(Context& context, SemIR::InstId inst_id) -> clang::QualType {
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

  clang::QualType mapped_type = MapNonWrapperType(context, inst_id, type_id);
  if (mapped_type.isNull()) {
    return mapped_type;
  }

  for (auto wrapper_type_id : llvm::reverse(wrapper_types)) {
    if (auto const_type = context.sem_ir().types().TryGetAs<SemIR::ConstType>(
            wrapper_type_id);
        const_type) {
      mapped_type.addConst();
    } else if (context.sem_ir().types().TryGetAs<SemIR::PointerType>(
                   wrapper_type_id)) {
      auto pointer_type = context.ast_context().getPointerType(mapped_type);
      mapped_type = context.ast_context().getAttributedType(
          clang::attr::TypeNonNull, pointer_type, pointer_type);
    } else {
      return clang::QualType();
    }
  }

  return mapped_type;
}

}  // namespace Carbon::Check
