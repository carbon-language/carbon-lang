// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp_type_mapping.h"

#include <cstddef>
#include <iostream>
#include <optional>

#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Lookup.h"
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
#include "toolchain/sem_ir/type_info.h"
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

// Attempts to look up a type by name, and returns the corresponding `QualType`,
// or a null type if lookup fails. `name_components` is the full path of the
// type, including any namespaces or nested types, separated into separate
// strings.
static auto LookupCppType(
    Context& context, std::initializer_list<llvm::StringRef> name_components)
    -> clang::QualType {
  clang::ASTUnit* ast = context.sem_ir().clang_ast_unit();
  CARBON_CHECK(ast);
  clang::Sema& sema = ast->getSema();

  clang::Decl* decl = sema.getASTContext().getTranslationUnitDecl();
  for (auto name_component : name_components) {
    auto* scope = dyn_cast<clang::DeclContext>(decl);
    if (!scope) {
      return clang::QualType();
    }

    // TODO: Map the LocId of the lookup to a clang SourceLocation and provide
    // it here so that clang's diagnostics can point into the carbon code that
    // uses the name.
    auto* identifier = sema.getPreprocessor().getIdentifierInfo(name_component);
    clang::LookupResult lookup(
        sema, clang::DeclarationNameInfo(identifier, clang::SourceLocation()),
        clang::Sema::LookupNameKind::LookupOrdinaryName);
    if (!sema.LookupQualifiedName(lookup, scope) || !lookup.isSingleResult()) {
      return clang::QualType();
    }
    decl = lookup.getFoundDecl();
  }

  auto* type_decl = dyn_cast<clang::TypeDecl>(decl);
  return type_decl ? sema.getASTContext().getTypeDeclType(type_decl)
                   : clang::QualType();
}

// Maps a Carbon class type to a C++ type. Returns a null `QualType` if the
// type is not supported.
static auto TryMapClassType(Context& context, SemIR::ClassType class_type)
    -> clang::QualType {
  // If the class was imported from C++, return the original C++ type.
  auto clang_decl_id =
      context.name_scopes()
          .Get(context.sem_ir().classes().Get(class_type.class_id).scope_id)
          .clang_decl_context_id();
  if (clang_decl_id.has_value()) {
    clang::Decl* clang_decl =
        context.sem_ir().clang_decls().Get(clang_decl_id).decl;
    auto* tag_type_decl = clang::cast<clang::TagDecl>(clang_decl);
    return context.ast_context().getCanonicalTagType(tag_type_decl);
  }

  // If the class represents a Carbon type literal, map it to the corresponding
  // C++ builtin type.
  auto literal = SemIR::TypeLiteralInfo::ForType(context.sem_ir(), class_type);
  switch (literal.kind) {
    case SemIR::TypeLiteralInfo::None: {
      break;
    }
    case SemIR::TypeLiteralInfo::Numeric: {
      switch (literal.numeric.kind) {
        case SemIR::NumericTypeLiteralInfo::None: {
          CARBON_FATAL("Unexpected invalid numeric type literal");
        }
        case SemIR::NumericTypeLiteralInfo::Float: {
          return context.ast_context().getRealTypeForBitwidth(
              literal.numeric.bit_width_id.AsValue(),
              clang::FloatModeKind::NoFloat);
        }
        case SemIR::NumericTypeLiteralInfo::Int: {
          return context.ast_context().getIntTypeForBitwidth(
              literal.numeric.bit_width_id.AsValue(), true);
        }
        case SemIR::NumericTypeLiteralInfo::UInt: {
          return context.ast_context().getIntTypeForBitwidth(
              literal.numeric.bit_width_id.AsValue(), false);
        }
      }
    }
    case SemIR::TypeLiteralInfo::Char: {
      return context.ast_context().CharTy;
    }
    case SemIR::TypeLiteralInfo::Str: {
      return LookupCppType(context, {"std", "string_view"});
    }
  }

  // Otherwise we don't have a mapping for this Carbon class type.
  // TODO: If the class type wasn't imported from C++, create a corresponding
  // C++ class type.
  return clang::QualType();
}

// Maps a non-wrapper (no const or pointer) Carbon type to a C++ type. Returns a
// null QualType if the type is not supported.
// TODO: Have both Carbon -> C++ and C++ -> Carbon mappings in a single place
// to keep them in sync.
static auto MapNonWrapperType(Context& context, SemIR::InstId inst_id,
                              SemIR::TypeId type_id) -> clang::QualType {
  auto type_inst = context.sem_ir().types().GetAsInst(type_id);

  CARBON_KIND_SWITCH(type_inst) {
    case SemIR::BoolType::Kind: {
      return context.ast_context().BoolTy;
    }
    case Carbon::SemIR::CharLiteralType::Kind: {
      return context.ast_context().CharTy;
    }
    case CARBON_KIND(SemIR::ClassType class_type): {
      return TryMapClassType(context, class_type);
    }
    case SemIR::IntLiteralType::Kind: {
      IntId bit_width_id = FindIntLiteralBitWidth(context, inst_id);
      if (bit_width_id == IntId::None) {
        return clang::QualType();
      }
      return context.ast_context().getIntTypeForBitwidth(bit_width_id.AsValue(),
                                                         true);
    }
    // TODO: What if the value doesn't fit to f64?
    case SemIR::FloatLiteralType::Kind: {
      return context.ast_context().DoubleTy;
    }
    default: {
      return clang::QualType();
    }
  }
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
