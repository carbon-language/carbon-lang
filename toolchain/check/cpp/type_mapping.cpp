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

// A function that wraps a C++ type to form another C++ type. Note that this is
// a raw function pointer; we don't currently use any lambda captures here. This
// can be replaced by a `std::function` if captures are found to be needed.
using WrapFn = auto (*)(Context& context, clang::QualType inner_type)
    -> clang::QualType;

// Represents a type that requires a subtype to be mapped into a Clang type
// before it can be mapped.
struct WrappedType {
  // The type contained in this wrapped type.
  SemIR::TypeId inner_type_id;
  // A function to construct the wrapped type from the mapped unwrapped type.
  WrapFn wrap_fn;
};

// Possible results from attempting to map a type. A null QualType indicates
// that the type couldn't be mapped.
using TryMapTypeResult = std::variant<clang::QualType, WrappedType>;

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
  auto arg = context.insts().TryGetAs<SemIR::IntValue>(
      context.constant_values().GetInstId(arg_const_id));
  if (!arg) {
    return IntId::None;
  }
  llvm::APInt arg_val = context.ints().Get(arg->int_id);
  int arg_non_sign_bits = arg_val.getSignificantBits() - 1;

  if (arg_non_sign_bits >= 128) {
    CARBON_DIAGNOSTIC(IntTooLargeForCppType, Error,
                      "integer value {0} too large to fit in a signed C++ "
                      "integer type; requires {1} bits, but max is 128",
                      TypedInt, int);
    context.emitter().Emit(arg_id, IntTooLargeForCppType,
                           {.type = arg->type_id, .value = arg_val},
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

// Returns the given integer type if its width is as expected. Otherwise returns
// the null type.
static auto VerifyIntegerTypeWidth(Context& context, clang::QualType type,
                                   unsigned int expected_width)
    -> clang::QualType {
  if (context.ast_context().getIntWidth(type) == expected_width) {
    return type;
  }
  return clang::QualType();
}

// Maps a Carbon class type to a C++ type. Returns a null `QualType` if the
// type is not supported.
static auto TryMapClassType(Context& context, SemIR::ClassType class_type)
    -> TryMapTypeResult {
  clang::ASTContext& ast_context = context.ast_context();

  // If the class was imported from C++, return the original C++ type.
  auto clang_decl_id =
      context.name_scopes()
          .Get(context.classes().Get(class_type.class_id).scope_id)
          .clang_decl_context_id();
  if (clang_decl_id.has_value()) {
    clang::Decl* clang_decl = context.clang_decls().Get(clang_decl_id).key.decl;
    auto* tag_type_decl = clang::cast<clang::TagDecl>(clang_decl);
    return ast_context.getCanonicalTagType(tag_type_decl);
  }

  // If the class represents a Carbon type literal, map it to the corresponding
  // C++ builtin type.
  auto type_info =
      SemIR::RecognizedTypeInfo::ForType(context.sem_ir(), class_type);
  switch (type_info.kind) {
    case SemIR::RecognizedTypeInfo::None: {
      break;
    }
    case SemIR::RecognizedTypeInfo::Numeric: {
      // Carbon supports large bit width beyond C++ builtins; we don't translate
      // those into integer types.
      if (!type_info.numeric.bit_width_id.is_embedded_value()) {
        break;
      }
      int bit_width = type_info.numeric.bit_width_id.AsValue();

      switch (type_info.numeric.kind) {
        case SemIR::NumericTypeLiteralInfo::None: {
          CARBON_FATAL("Unexpected invalid numeric type literal");
        }
        case SemIR::NumericTypeLiteralInfo::Float: {
          return ast_context.getRealTypeForBitwidth(
              bit_width, clang::FloatModeKind::NoFloat);
        }
        case SemIR::NumericTypeLiteralInfo::Int: {
          return ast_context.getIntTypeForBitwidth(bit_width, true);
        }
        case SemIR::NumericTypeLiteralInfo::UInt: {
          return ast_context.getIntTypeForBitwidth(bit_width, false);
        }
      }
    }
    case SemIR::RecognizedTypeInfo::Char: {
      return ast_context.CharTy;
    }
    case SemIR::RecognizedTypeInfo::CppLong32: {
      return VerifyIntegerTypeWidth(context, ast_context.LongTy, 32);
    }
    case SemIR::RecognizedTypeInfo::CppULong32: {
      return VerifyIntegerTypeWidth(context, ast_context.UnsignedLongTy, 32);
    }
    case SemIR::RecognizedTypeInfo::CppLongLong64: {
      return VerifyIntegerTypeWidth(context, ast_context.LongLongTy, 64);
    }
    case SemIR::RecognizedTypeInfo::CppULongLong64: {
      return VerifyIntegerTypeWidth(context, ast_context.UnsignedLongLongTy,
                                    64);
    }
    case SemIR::RecognizedTypeInfo::CppNullptrT: {
      return ast_context.NullPtrTy;
    }
    case SemIR::RecognizedTypeInfo::CppVoidBase: {
      return ast_context.VoidTy;
    }
    case SemIR::RecognizedTypeInfo::Optional: {
      auto args = context.inst_blocks().GetOrEmpty(type_info.args_id);
      if (args.size() == 1) {
        auto arg_id = args[0];
        if (auto facet = context.insts().TryGetAs<SemIR::FacetValue>(arg_id)) {
          arg_id = facet->type_inst_id;
        }
        if (auto pointer_type =
                context.insts().TryGetAs<SemIR::PointerType>(arg_id)) {
          return WrappedType{
              .inner_type_id = context.types().GetTypeIdForTypeInstId(
                  pointer_type->pointee_id),
              .wrap_fn = [](Context& context, clang::QualType inner_type) {
                return context.ast_context().getPointerType(inner_type);
              }};
        }
      }
      break;
    }
    case SemIR::RecognizedTypeInfo::Str: {
      return LookupCppType(context, {"std", "string_view"});
    }
  }

  // Otherwise we don't have a mapping for this Carbon class type.
  // TODO: If the class type wasn't imported from C++, create a corresponding
  // C++ class type.
  return clang::QualType();
}

// Maps a Carbon type to a C++ type. Either returns the mapped type, a null type
// as a placeholder indicating the type can't be mapped, or a `WrappedType`
// representing a type that needs more work before it can be mapped.
// TODO: Have both Carbon -> C++ and C++ -> Carbon mappings in a single place
// to keep them in sync.
static auto TryMapType(Context& context, SemIR::TypeId type_id)
    -> TryMapTypeResult {
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
    case CARBON_KIND(SemIR::ConstType const_type): {
      return WrappedType{
          .inner_type_id =
              context.types().GetTypeIdForTypeInstId(const_type.inner_id),
          .wrap_fn = [](Context& /*context*/, clang::QualType inner_type) {
            return inner_type.withConst();
          }};
    }
    case SemIR::FloatLiteralType::Kind: {
      return context.ast_context().DoubleTy;
    }
    case CARBON_KIND(SemIR::PointerType pointer_type): {
      return WrappedType{
          .inner_type_id =
              context.types().GetTypeIdForTypeInstId(pointer_type.pointee_id),
          .wrap_fn = [](Context& context, clang::QualType inner_type) {
            auto pointer_type =
                context.ast_context().getPointerType(inner_type);
            return context.ast_context().getAttributedType(
                clang::attr::TypeNonNull, pointer_type, pointer_type);
          }};
    }

    default: {
      return clang::QualType();
    }
  }

  return clang::QualType();
}

auto MapToCppType(Context& context, SemIR::TypeId type_id) -> clang::QualType {
  // TODO: unify this with the C++ to Carbon type mapping function.
  llvm::SmallVector<WrapFn> wrap_fns;
  while (true) {
    CARBON_KIND_SWITCH(TryMapType(context, type_id)) {
      case CARBON_KIND(clang::QualType type): {
        for (auto wrap_fn : llvm::reverse(wrap_fns)) {
          if (type.isNull()) {
            break;
          }
          type = wrap_fn(context, type);
        }
        return type;
      }

      case CARBON_KIND(WrappedType wrapped): {
        wrap_fns.push_back(wrapped.wrap_fn);
        type_id = wrapped.inner_type_id;
        break;
      }
    }
  }
}

auto InventClangArg(Context& context, SemIR::InstId arg_id) -> clang::Expr* {
  clang::ExprValueKind value_kind;
  switch (SemIR::GetExprCategory(context.sem_ir(), arg_id)) {
    case SemIR::ExprCategory::Error:
      return nullptr;

    case SemIR::ExprCategory::Pattern:
      CARBON_FATAL("Passing a pattern as a function argument");

    case SemIR::ExprCategory::DurableRef:
    case SemIR::ExprCategory::RefTagged:
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

  clang::QualType arg_cpp_type;

  // Special case: if the argument is an integer literal, look at its value.
  // TODO: Consider producing a `clang::IntegerLiteral` in this case instead, so
  // that C++ overloads that behave differently for zero-valued int literals can
  // recognize it.
  auto type_id = context.insts().Get(arg_id).type_id();
  if (context.types().Is<SemIR::IntLiteralType>(type_id)) {
    IntId bit_width_id = FindIntLiteralBitWidth(context, arg_id);
    if (bit_width_id != IntId::None) {
      arg_cpp_type = context.ast_context().getIntTypeForBitwidth(
          bit_width_id.AsValue(), true);
    }
  }

  if (arg_cpp_type.isNull()) {
    arg_cpp_type = MapToCppType(context, type_id);
  }

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
