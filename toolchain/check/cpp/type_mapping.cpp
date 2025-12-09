// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/type_mapping.h"

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
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/literal.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/type.h"
#include "toolchain/sem_ir/type_info.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Find the bit width of an integer literal. Following the C++ standard rules
// for assigning a type to a decimal integer literal, the first signed integer
// in which the value could fit among bit widths of 32, 64 and 128 is selected.
// If the value can't fit into a signed integer with width of 128-bits, then a
// diagnostic is emitted and the function returns IntId::None. Returns
// IntId::None also if the argument is not a constant integer, if it is an
// error constant, or if it is a symbolic constant.
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
  llvm::APInt arg_val = context.ints().Get(arg.int_id);
  int arg_non_sign_bits = arg_val.getSignificantBits() - 1;

  if (arg_non_sign_bits >= 128) {
    CARBON_DIAGNOSTIC(IntTooLargeForCppType, Error,
                      "integer value {0} too large to fit in a signed C++ "
                      "integer type; requires {1} bits, but max is 128",
                      TypedInt, int);
    context.emitter().Emit(arg_id, IntTooLargeForCppType,
                           {.type = arg.type_id, .value = arg_val},
                           arg_non_sign_bits + 1);
    return IntId::None;
  }

  return (arg_non_sign_bits < 32)
             ? IntId::MakeRaw(32)
             : ((arg_non_sign_bits < 64) ? IntId::MakeRaw(64)
                                         : IntId::MakeRaw(128));
}

// Attempts to look up a type by name, and returns the corresponding `QualType`,
// or a null type if lookup fails. `name_components` is the full path of the
// type, including any namespaces or nested types, separated into separate
// strings.
static auto LookupCppType(
    Context& context, std::initializer_list<llvm::StringRef> name_components)
    -> clang::QualType {
  clang::Sema& sema = context.clang_sema();

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
          .Get(context.classes().Get(class_type.class_id).scope_id)
          .clang_decl_context_id();
  if (clang_decl_id.has_value()) {
    clang::Decl* clang_decl = context.clang_decls().Get(clang_decl_id).key.decl;
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
      // Carbon supports large bit width beyond C++ builtins; we don't need to
      // translate those.
      if (!literal.numeric.bit_width_id.is_embedded_value()) {
        return clang::QualType();
      }
      int bit_width = literal.numeric.bit_width_id.AsValue();

      switch (literal.numeric.kind) {
        case SemIR::NumericTypeLiteralInfo::None: {
          CARBON_FATAL("Unexpected invalid numeric type literal");
        }
        case SemIR::NumericTypeLiteralInfo::Float: {
          return context.ast_context().getRealTypeForBitwidth(
              bit_width, clang::FloatModeKind::NoFloat);
        }
        case SemIR::NumericTypeLiteralInfo::Int: {
          return context.ast_context().getIntTypeForBitwidth(bit_width, true);
        }
        case SemIR::NumericTypeLiteralInfo::UInt: {
          return context.ast_context().getIntTypeForBitwidth(bit_width, false);
        }
      }
    }
    case SemIR::TypeLiteralInfo::Char: {
      return context.ast_context().CharTy;
    }
    case SemIR::TypeLiteralInfo::CppNullptrT: {
      return context.ast_context().NullPtrTy;
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
  auto type_inst = context.types().GetAsInst(type_id);

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
    case SemIR::FloatLiteralType::Kind: {
      return context.ast_context().DoubleTy;
    }
    default: {
      return clang::QualType();
    }
  }
}

// Returns `void*` if the type is a wrapped `Cpp.void*`, consuming the pointer
// from `wrapper_types`. Otherwise returns no type.
static auto TryMapVoidPointer(Context& context, SemIR::TypeId type_id,
                              llvm::SmallVector<SemIR::TypeId>& wrapper_types)
    -> clang::QualType {
  if (type_id != SemIR::CppVoidType::TypeId || wrapper_types.empty()) {
    return clang::QualType();
  }

  if (context.types().Is<SemIR::PointerType>(wrapper_types.back())) {
    // `void*`.
    wrapper_types.pop_back();
  } else if (wrapper_types.size() >= 2 &&
             context.types().Is<SemIR::ConstType>(wrapper_types.back()) &&
             context.types().Is<SemIR::PointerType>(
                 wrapper_types[wrapper_types.size() - 2])) {
    // `const void*`.
    wrapper_types.erase(wrapper_types.end() - 2);
  } else {
    return clang::QualType();
  }

  return context.ast_context().getAttributedType(
      clang::attr::TypeNonNull, context.ast_context().VoidPtrTy,
      context.ast_context().VoidPtrTy);
}

// Maps a Carbon type to a C++ type. Accepts an InstId, representing a value
// whose type is mapped to a C++ type. Returns `clang::QualType` if the mapping
// succeeds, or `clang::QualType::isNull()` if the type is not supported.
// TODO: unify this with the C++ to Carbon type mapping function.
static auto MapToCppType(Context& context, SemIR::InstId inst_id)
    -> clang::QualType {
  auto type_id = context.insts().Get(inst_id).type_id();
  llvm::SmallVector<SemIR::TypeId> wrapper_types;
  while (true) {
    SemIR::TypeId orig_type_id = type_id;
    if (auto const_type = context.types().TryGetAs<SemIR::ConstType>(type_id);
        const_type) {
      type_id = context.types().GetTypeIdForTypeInstId(const_type->inner_id);
    } else if (auto pointer_type =
                   context.types().TryGetAs<SemIR::PointerType>(type_id);
               pointer_type) {
      type_id =
          context.types().GetTypeIdForTypeInstId(pointer_type->pointee_id);
    } else {
      break;
    }
    wrapper_types.push_back(orig_type_id);
  }

  clang::QualType mapped_type =
      TryMapVoidPointer(context, type_id, wrapper_types);
  if (mapped_type.isNull()) {
    mapped_type = MapNonWrapperType(context, inst_id, type_id);
    if (mapped_type.isNull()) {
      return mapped_type;
    }
  }

  for (auto wrapper_type_id : llvm::reverse(wrapper_types)) {
    if (auto const_type =
            context.types().TryGetAs<SemIR::ConstType>(wrapper_type_id);
        const_type) {
      mapped_type.addConst();
    } else if (context.types().TryGetAs<SemIR::PointerType>(wrapper_type_id)) {
      auto pointer_type = context.ast_context().getPointerType(mapped_type);
      mapped_type = context.ast_context().getAttributedType(
          clang::attr::TypeNonNull, pointer_type, pointer_type);
    } else {
      return clang::QualType();
    }
  }

  return mapped_type;
}

auto InventClangArg(Context& context, SemIR::InstId arg_id) -> clang::Expr* {
  clang::ExprValueKind value_kind;
  switch (SemIR::GetExprCategory(context.sem_ir(), arg_id)) {
    case SemIR::ExprCategory::Error:
      return nullptr;

    case SemIR::ExprCategory::DurableRef:
      value_kind = clang::ExprValueKind::VK_LValue;
      break;

    case SemIR::ExprCategory::EphemeralRef:
      value_kind = clang::ExprValueKind::VK_XValue;
      break;

    case SemIR::ExprCategory::NotExpr:
      // A call using this expression as an argument won't be valid, but we
      // don't diagnose that until we convert the expression to the parameter
      // type.
      value_kind = clang::ExprValueKind::VK_PRValue;
      break;

    case SemIR::ExprCategory::Value:
    case SemIR::ExprCategory::Initializing:
      value_kind = clang::ExprValueKind::VK_PRValue;
      break;

    case SemIR::ExprCategory::Mixed:
      // TODO: Handle this by creating an InitListExpr.
      value_kind = clang::ExprValueKind::VK_PRValue;
      break;
  }

  if (context.insts().Get(arg_id).type_id() == SemIR::ErrorInst::TypeId) {
    // The argument error has already been diagnosed.
    return nullptr;
  }

  clang::QualType arg_cpp_type = MapToCppType(context, arg_id);
  if (arg_cpp_type.isNull()) {
    CARBON_DIAGNOSTIC(CppCallArgTypeNotSupported, Error,
                      "call argument of type {0} is not supported",
                      TypeOfInstId);
    context.emitter().Emit(arg_id, CppCallArgTypeNotSupported, arg_id);
    return nullptr;
  }

  // TODO: Avoid heap allocating more of these on every call. Either cache them
  // somewhere or put them on the stack.
  return new (context.ast_context())
      clang::OpaqueValueExpr(GetCppLocation(context, SemIR::LocId(arg_id)),
                             arg_cpp_type.getNonReferenceType(), value_kind);
}

auto InventClangArgs(Context& context, llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<llvm::SmallVector<clang::Expr*>> {
  llvm::SmallVector<clang::Expr*> arg_exprs;
  arg_exprs.reserve(arg_ids.size());
  for (SemIR::InstId arg_id : arg_ids) {
    auto* arg_expr = InventClangArg(context, arg_id);
    if (!arg_expr) {
      return std::nullopt;
    }
    arg_exprs.push_back(arg_expr);
  }
  return arg_exprs;
}

}  // namespace Carbon::Check
