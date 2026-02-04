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
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/int.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/literal.h"
#include "toolchain/sem_ir/class.h"
#include "toolchain/sem_ir/expr_info.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/inst_kind.h"
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
static auto FindIntLiteralBitWidth(Context& context, SemIR::LocId loc_id,
                                   SemIR::ConstantId arg_const_id) -> IntId {
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
    context.emitter().Emit(loc_id, IntTooLargeForCppType,
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

namespace {
// Information about the form of an expression.
struct FormInfo {
  enum Kind : int8_t {
    Primitive,
    Tuple,
    Struct,
  };

  // The kind of the form.
  Kind kind;
  // The category component of the form. For a composite form, if this is not
  // `Mixed` it represents the category component of all primitive sub-forms
  // of this form.
  SemIR::ExprCategory category;
  // The type component of the form.
  SemIR::TypeId type_id;
  // The constant value component of the form.
  SemIR::ConstantId constant_id;
  // The location of the expression whose form this is.
  SemIR::LocId loc_id;
  // The underlying instruction, if there is one. This is only present in order
  // to support lazy form decomposition, and should not be used for other
  // purposes. May be `None` if this is not the form of a tuple or struct
  // literal that can be decomposed further.
  SemIR::InstId inst_id;

  // Returns whether this is a compound form.
  auto is_compound() const -> bool { return kind == Tuple || kind == Struct; }
};
}  // namespace

// Given a type, determines the category of the decomposed form of an expression
// of that type. This is Primitive if the type does not support form
// decomposition.
static auto GetDecomposedFormKindForType(Context& context,
                                         SemIR::TypeId type_id)
    -> FormInfo::Kind {
  if (context.types().Is<SemIR::TupleType>(type_id)) {
    return FormInfo::Tuple;
  }
  if (context.types().Is<SemIR::StructType>(type_id)) {
    return FormInfo::Struct;
  }
  return FormInfo::Primitive;
}

// Gets information about the form of an instruction.
static auto GetFormInfo(Context& context, SemIR::InstId inst_id) -> FormInfo {
  auto inst = context.insts().Get(inst_id);

  SemIR::ExprCategory category =
      SemIR::GetExprCategory(context.sem_ir(), inst_id);
  if (inst.type_id() == SemIR::ErrorInst::TypeId) {
    // TODO: Should `GetExprCategory` do this?
    category = SemIR::ExprCategory::Error;
  }

  FormInfo::Kind kind = FormInfo::Primitive;
  if (category == SemIR::ExprCategory::Mixed) {
    kind = GetDecomposedFormKindForType(context, inst.type_id());
    CARBON_CHECK(kind != FormInfo::Primitive,
                 "Unexpected type {0} for mixed category",
                 context.types().GetAsInst(inst.type_id()));
  }

  return {.kind = kind,
          .category = category,
          .type_id = inst.type_id(),
          .constant_id = context.constant_values().Get(inst_id),
          .loc_id = SemIR::LocId(inst_id),
          .inst_id = inst_id};
}

// Given a form, attempts to perform form decomposition, converting it from a
// primitive form into a compound form if possible. Otherwise, returns the form
// unchanged.
static auto DecomposeForm(Context& context, FormInfo form) -> FormInfo {
  if (form.kind == FormInfo::Primitive) {
    form.kind = GetDecomposedFormKindForType(context, form.type_id);
    // TODO: Should we replace a category of Initializing with
    // EphemeralReference here to model temporary materialization if we
    // performed decomposition?
  }
  return form;
}

using FormVisitor = llvm::function_ref<auto(FormInfo)->void>;

// Gets information about the forms of the instructions in a block.
static auto VisitFormInfos(Context& context, SemIR::InstBlockId inst_block_id,
                           FormVisitor visitor) -> void {
  auto inst_ids = context.inst_blocks().Get(inst_block_id);
  for (auto inst_id : inst_ids) {
    visitor(GetFormInfo(context, inst_id));
  }
}

// Given a tuple form, visits the forms of the elements.
static auto VisitTupleElementForms(Context& context, FormInfo form,
                                   FormVisitor visitor) -> void {
  // If we have a tuple literal, directly grab the forms of its elements.
  if (auto tuple_lit_inst =
          context.insts().TryGetAsIfValid<SemIR::TupleLiteral>(form.inst_id)) {
    VisitFormInfos(context, tuple_lit_inst->elements_id, visitor);
    return;
  }

  // Otherwise, decompose the type and, if available, the constant value.
  auto tuple_type = context.types().GetAs<SemIR::TupleType>(form.type_id);
  auto element_type_inst_ids =
      context.inst_blocks().Get(tuple_type.type_elements_id);

  llvm::SmallVector<FormInfo> result;
  result.reserve(element_type_inst_ids.size());

  auto tuple_const_inst = context.insts().TryGetAsIfValid<SemIR::TupleValue>(
      context.constant_values().GetInstIdIfValid(form.constant_id));
  auto tuple_const_inst_ids =
      tuple_const_inst
          ? context.inst_blocks().Get(tuple_const_inst->elements_id)
          : llvm::ArrayRef<SemIR::InstId>();

  for (auto [type_inst_id, const_inst_id] :
       llvm::zip_longest(element_type_inst_ids, tuple_const_inst_ids)) {
    visitor({.kind = FormInfo::Primitive,
             .category = form.category,
             .type_id = context.types().GetTypeIdForTypeInstId(*type_inst_id),
             .constant_id = const_inst_id
                                ? context.constant_values().Get(*const_inst_id)
                                : SemIR::ConstantId::NotConstant,
             .loc_id = form.loc_id,
             .inst_id = SemIR::InstId::None});
  }
}

// Given a struct form, returns the forms of the elements.
static auto VisitStructElementForms(Context& context, FormInfo form,
                                    FormVisitor visitor) -> void {
  // If we have a struct literal, directly grab the forms of its elements.
  if (auto struct_lit_inst =
          context.insts().TryGetAsIfValid<SemIR::StructLiteral>(form.inst_id)) {
    VisitFormInfos(context, struct_lit_inst->elements_id, visitor);
    return;
  }

  // Otherwise, decompose the type and, if available, the constant value.
  auto struct_type = context.types().GetAs<SemIR::StructType>(form.type_id);
  auto fields = context.struct_type_fields().Get(struct_type.fields_id);

  llvm::SmallVector<FormInfo> result;
  result.reserve(fields.size());

  auto struct_const_inst = context.insts().TryGetAsIfValid<SemIR::StructValue>(
      context.constant_values().GetInstIdIfValid(form.constant_id));
  auto struct_const_inst_ids =
      struct_const_inst
          ? context.inst_blocks().Get(struct_const_inst->elements_id)
          : llvm::ArrayRef<SemIR::InstId>();

  for (auto [field, const_inst_id] :
       llvm::zip_longest(fields, struct_const_inst_ids)) {
    visitor(
        {.kind = FormInfo::Primitive,
         .category = form.category,
         .type_id = context.types().GetTypeIdForTypeInstId(field->type_inst_id),
         .constant_id = const_inst_id
                            ? context.constant_values().Get(*const_inst_id)
                            : SemIR::ConstantId::NotConstant,
         .loc_id = form.loc_id,
         .inst_id = SemIR::InstId::None});
  }
}

// Invent a primitive Clang argument given the form of the corresponding Carbon
// expression.
static auto InventPrimitiveClangArg(Context& context, FormInfo form)
    -> clang::Expr* {
  clang::ExprValueKind value_kind;
  switch (form.category) {
    case SemIR::ExprCategory::Error:
      // The argument error has already been diagnosed.
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
    case SemIR::ExprCategory::ReprInitializing:
    case SemIR::ExprCategory::InPlaceInitializing:
      value_kind = clang::ExprValueKind::VK_PRValue;
      break;

    case SemIR::ExprCategory::Mixed:
      CARBON_FATAL("Argument does not have primitive form");
  }

  clang::QualType arg_cpp_type;

  // Special case: if the argument is an integer literal, look at its value.
  // TODO: Consider producing a `clang::IntegerLiteral` in this case instead, so
  // that C++ overloads that behave differently for zero-valued int literals can
  // recognize it.
  if (context.types().Is<SemIR::IntLiteralType>(form.type_id)) {
    IntId bit_width_id =
        FindIntLiteralBitWidth(context, form.loc_id, form.constant_id);
    if (bit_width_id != IntId::None) {
      arg_cpp_type = context.ast_context().getIntTypeForBitwidth(
          bit_width_id.AsValue(), true);
    }
  }

  if (arg_cpp_type.isNull()) {
    arg_cpp_type = MapToCppType(context, form.type_id);
  }

  if (arg_cpp_type.isNull()) {
    CARBON_DIAGNOSTIC(CppCallArgTypeNotSupported, Error,
                      "call argument of type {0} is not supported",
                      SemIR::TypeId);
    context.emitter().Emit(form.loc_id, CppCallArgTypeNotSupported,
                           form.type_id);
    return nullptr;
  }

  // TODO: Avoid heap allocating more of these on every call. Either cache them
  // somewhere or put them on the stack.
  return new (context.ast_context())
      clang::OpaqueValueExpr(GetCppLocation(context, form.loc_id),
                             arg_cpp_type.getNonReferenceType(), value_kind);
}

// Invent an initializer list Clang argument given the form of the corresponding
// Carbon expression, which is a compound form. The initializers for the
// elements are taken from the end of `results`.
static auto InventCompoundClangArg(Context& context, FormInfo form,
                                   llvm::SmallVectorImpl<clang::Expr*>& results)
    -> clang::Expr* {
  auto make_init_list = [&](llvm::ArrayRef<clang::Expr*> inits) {
    // TODO: Compute the `(` and `)` locations for a tuple literal or the `{`
    // and `}` locations for a struct literal.
    auto compound_loc = GetCppLocation(context, form.loc_id);
    auto lbrace_loc = compound_loc;
    auto rbrace_loc = compound_loc;

    auto* init_list = new (context.ast_context()) clang::InitListExpr(
        context.ast_context(), lbrace_loc, inits, rbrace_loc);
    init_list->setType(context.ast_context().VoidTy);
    return init_list;
  };

  switch (form.kind) {
    case FormInfo::Primitive:
      CARBON_FATAL("Not a compound form");

    case FormInfo::Tuple: {
      // For a tuple, form a non-designated init list containing the
      // corresponding initializers.
      auto num_elements = context.inst_blocks()
                              .Get(context.types()
                                       .GetAs<SemIR::TupleType>(form.type_id)
                                       .type_elements_id)
                              .size();
      CARBON_CHECK(results.size() >= num_elements);

      auto* init_list =
          make_init_list(llvm::ArrayRef(results).take_back(num_elements));
      results.truncate(results.size() - num_elements);
      return init_list;
    }

    case FormInfo::Struct: {
      // For a struct, form a designated initializer list, converting the struct
      // field names into designator names.
      auto fields = context.struct_type_fields().Get(
          context.types().GetAs<SemIR::StructType>(form.type_id).fields_id);
      llvm::SmallVector<clang::Expr*> field_inits;
      field_inits.reserve(fields.size());

      for (auto [field, init] : llvm::zip(
               fields, llvm::ArrayRef(results).take_back(fields.size()))) {
        auto loc = init->getExprLoc();
        auto* field_name = GetClangIdentifierInfo(context, field.name_id);
        if (!field_name) {
          CARBON_DIAGNOSTIC(CppCallFieldNameNotSupported, Error,
                            "field name `{0}` cannot be mapped into C++",
                            SemIR::NameId);
          context.emitter().Emit(form.loc_id, CppCallFieldNameNotSupported,
                                 field.name_id);
          return nullptr;
        }

        auto designator =
            clang::DesignatedInitExpr::Designator::CreateFieldDesignator(
                field_name, /*DotLoc=*/loc, /*FieldLoc=*/loc);
        field_inits.push_back(clang::DesignatedInitExpr::Create(
            context.ast_context(), designator, /*IndexExprs*/ {},
            /*EqualOrColonLoc=*/loc, /*GNUSyntax=*/false, init));
      }

      results.truncate(results.size() - fields.size());

      return make_init_list(field_inits);
    }
  }
}

auto InventClangArg(Context& context, SemIR::InstId arg_id) -> clang::Expr* {
  enum Phase { Initial, AfterSubexpressions };
  llvm::SmallVector<std::pair<FormInfo, Phase>> worklist = {
      {GetFormInfo(context, arg_id), Initial}};
  llvm::SmallVector<clang::Expr*> pending_results = {};

  while (!worklist.empty()) {
    auto [form, phase] = worklist.pop_back_val();

    switch (phase) {
      case Initial: {
        form = DecomposeForm(context, form);
        switch (form.kind) {
          case FormInfo::Primitive: {
            auto* expr = InventPrimitiveClangArg(context, form);
            if (!expr) {
              return nullptr;
            }
            pending_results.push_back(expr);
            break;
          }

          case FormInfo::Tuple:
          case FormInfo::Struct: {
            worklist.push_back({form, AfterSubexpressions});
            auto initial_size = worklist.size();
            auto visitor = [&](FormInfo element) {
              worklist.push_back({element, Initial});
            };
            if (form.kind == FormInfo::Tuple) {
              VisitTupleElementForms(context, form, visitor);
            } else {
              VisitStructElementForms(context, form, visitor);
            }
            // Reverse the added elements so that we pop them in element order.
            std::reverse(worklist.begin() + initial_size, worklist.end());
            break;
          }
        }
        break;
      }

      case AfterSubexpressions: {
        auto* expr = InventCompoundClangArg(context, form, pending_results);
        if (!expr) {
          return nullptr;
        }
        pending_results.push_back(expr);
        break;
      }
    }
  }

  CARBON_CHECK(pending_results.size() == 1);
  return pending_results.back();
}

auto InventClangArgs(Context& context, llvm::ArrayRef<SemIR::InstId> arg_ids)
    -> std::optional<llvm::SmallVector<clang::Expr*>> {
  std::optional<llvm::SmallVector<clang::Expr*>> arg_exprs;
  arg_exprs.emplace();
  arg_exprs->reserve(arg_ids.size());
  for (SemIR::InstId arg_id : arg_ids) {
    auto* arg_expr = InventClangArg(context, arg_id);
    if (!arg_expr) {
      arg_exprs = std::nullopt;
      return arg_exprs;
    }
    arg_exprs->push_back(arg_expr);
  }
  return arg_exprs;
}

}  // namespace Carbon::Check
