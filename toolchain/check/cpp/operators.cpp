// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/operators.h"

#include "clang/Sema/Initialization.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Sema.h"
#include "toolchain/check/convert.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/check/cpp/import.h"
#include "toolchain/check/cpp/location.h"
#include "toolchain/check/cpp/overload_resolution.h"
#include "toolchain/check/cpp/type_mapping.h"
#include "toolchain/check/custom_witness.h"
#include "toolchain/check/function.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/pattern.h"
#include "toolchain/check/type.h"
#include "toolchain/check/type_completion.h"
#include "toolchain/sem_ir/builtin_function_kind.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/cpp_initializer_list.h"
#include "toolchain/sem_ir/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Maps Carbon operator interface and operator names to Clang operator kinds.
static auto GetClangOperatorKind(Context& context, SemIR::LocId loc_id,
                                 CoreIdentifier interface_name,
                                 CoreIdentifier op_name)
    -> std::optional<clang::OverloadedOperatorKind> {
  switch (interface_name) {
      // Unary operators.
    case CoreIdentifier::Destroy:
    case CoreIdentifier::As:
    case CoreIdentifier::ImplicitAs:
    case CoreIdentifier::UnsafeAs:
    case CoreIdentifier::Copy: {
      // TODO: Support destructors and conversions.
      return std::nullopt;
    }

    // Increment and decrement.
    case CoreIdentifier::Inc: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PlusPlus;
    }
    case CoreIdentifier::Dec: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_MinusMinus;
    }

    // Arithmetic.
    case CoreIdentifier::Negate: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Minus;
    }

    // Bitwise.
    case CoreIdentifier::BitComplement: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Tilde;
    }

    // Binary operators.

    // Arithmetic operators.
    case CoreIdentifier::AddWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Plus;
    }
    case CoreIdentifier::SubWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Minus;
    }
    case CoreIdentifier::MulWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Star;
    }
    case CoreIdentifier::DivWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Slash;
    }
    case CoreIdentifier::ModWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Percent;
    }

    // Bitwise operators.
    case CoreIdentifier::BitAndWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Amp;
    }
    case CoreIdentifier::BitOrWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Pipe;
    }
    case CoreIdentifier::BitXorWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Caret;
    }
    case CoreIdentifier::LeftShiftWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_LessLess;
    }
    case CoreIdentifier::RightShiftWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_GreaterGreater;
    }

    // Assignment.
    case CoreIdentifier::AssignWith: {
      // TODO: This is not yet reached because we don't use the `AssignWith`
      // interface for assignment yet.
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_Equal;
    }

    // Compound assignment arithmetic operators.
    case CoreIdentifier::AddAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PlusEqual;
    }
    case CoreIdentifier::SubAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_MinusEqual;
    }
    case CoreIdentifier::MulAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_StarEqual;
    }
    case CoreIdentifier::DivAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_SlashEqual;
    }
    case CoreIdentifier::ModAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PercentEqual;
    }

    // Compound assignment bitwise operators.
    case CoreIdentifier::BitAndAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_AmpEqual;
    }
    case CoreIdentifier::BitOrAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_PipeEqual;
    }
    case CoreIdentifier::BitXorAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_CaretEqual;
    }
    case CoreIdentifier::LeftShiftAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_LessLessEqual;
    }
    case CoreIdentifier::RightShiftAssignWith: {
      CARBON_CHECK(op_name == CoreIdentifier::Op);
      return clang::OO_GreaterGreaterEqual;
    }

    // Relational operators.
    case CoreIdentifier::EqWith: {
      if (op_name == CoreIdentifier::Equal) {
        return clang::OO_EqualEqual;
      }
      CARBON_CHECK(op_name == CoreIdentifier::NotEqual);
      return clang::OO_ExclaimEqual;
    }
    case CoreIdentifier::OrderedWith: {
      switch (op_name) {
        case CoreIdentifier::Less:
          return clang::OO_Less;
        case CoreIdentifier::Greater:
          return clang::OO_Greater;
        case CoreIdentifier::LessOrEquivalent:
          return clang::OO_LessEqual;
        case CoreIdentifier::GreaterOrEquivalent:
          return clang::OO_GreaterEqual;
        default:
          CARBON_FATAL("Unexpected OrderedWith op `{0}`", op_name);
      }
    }

    // Array indexing.
    case CoreIdentifier::IndexWith: {
      CARBON_CHECK(op_name == CoreIdentifier::At);
      return clang::OO_Subscript;
    }

    default: {
      context.TODO(loc_id, llvm::formatv("Unsupported operator interface `{0}`",
                                         interface_name));
      return std::nullopt;
    }
  }
}

// Creates and returns a function that can be used to construct a
// std::initializer_list from an array.
//
// TODO: This should ideally be implemented in Carbon code rather than by
// synthesizing a function.
// TODO: We should cache and reuse the generated function.
static auto MakeCppStdInitializerListMake(Context& context, SemIR::LocId loc_id,
                                          clang::QualType init_list_type,
                                          int32_t size) -> SemIR::InstId {
  // Extract the element type `T` from the `std::initializer_list<T>` type.
  clang::QualType element_type;
  bool is_std_initializer_list =
      context.clang_sema().isStdInitializerList(init_list_type, &element_type);
  CARBON_CHECK(is_std_initializer_list);
  auto element_type_inst_id =
      ImportCppType(context, loc_id, element_type).inst_id;
  if (element_type_inst_id == SemIR::ErrorInst::InstId) {
    return SemIR::ErrorInst::InstId;
  }

  // Import the `std::initializer_list<T>` type and check we recognize its
  // layout.
  auto [init_list_type_inst_id, init_list_type_id] =
      ImportCppType(context, loc_id, init_list_type);
  if (init_list_type_id == SemIR::ErrorInst::TypeId) {
    return SemIR::ErrorInst::InstId;
  }
  auto layout =
      SemIR::GetStdInitializerListLayout(context.sem_ir(), init_list_type_id);
  if (layout.kind == SemIR::StdInitializerListLayout::None) {
    context.TODO(loc_id, "Unsupported layout for std::initializer_list");
    return SemIR::ErrorInst::InstId;
  }
  auto init_list_class_id = context.sem_ir()
                                .types()
                                .GetAs<SemIR::ClassType>(init_list_type_id)
                                .class_id;
  auto& init_list_class = context.classes().Get(init_list_class_id);

  // Build the array type `T[size]` that we use as the parameter type.
  // TODO: This will eventually be called from impl lookup, possibly while
  // forming a specific, so we should not be adding instructions here.
  auto bound_id = AddInst(
      context, SemIR::LocIdAndInst(
                   loc_id, SemIR::IntValue{
                               .type_id = GetSingletonType(
                                   context, SemIR::IntLiteralType::TypeInstId),
                               .int_id = context.ints().Add(size)}));
  auto array_type_inst_id = AddTypeInst(
      context,
      SemIR::LocIdAndInst::RuntimeVerified(
          context.sem_ir(), loc_id,
          SemIR::ArrayType{.type_id = SemIR::TypeType::TypeId,
                           .bound_id = bound_id,
                           .element_type_inst_id = element_type_inst_id}));
  auto array_type_id =
      context.types().GetTypeIdForTypeInstId(array_type_inst_id);

  // Create a builtin function to perform the conversion from array type to
  // initializer list type. We name the synthesized function as if it were a
  // constructor of std::initializer_list.
  // TODO: Find a better way to handle this. Ideally we should stop using this
  // function entirely and declare the necessary builtin in the prelude.
  auto [decl_id, function_id] =
      MakeGeneratedFunctionDecl(context, loc_id,
                                {.parent_scope_id = init_list_class.scope_id,
                                 .name_id = init_list_class.name_id,
                                 .param_type_ids = {array_type_id},
                                 .param_kind = ParamPatternKind::Value,
                                 .return_type_id = init_list_type_id});

  auto& function = context.functions().Get(function_id);
  CARBON_CHECK(IsValidBuiltinDeclaration(
      context, function,
      SemIR::BuiltinFunctionKind::CppStdInitializerListMake));
  function.SetBuiltinFunction(
      SemIR::BuiltinFunctionKind::CppStdInitializerListMake);

  return decl_id;
}

// Returns information about the Carbon signature to import when importing a C++
// constructor or conversion operator.
static auto GetConversionSignatureToImport(
    Context& context, SemIR::InstId source_id,
    clang::InitializationSequence::StepKind step_kind,
    clang::FunctionDecl* function_decl, clang::DeclAccessPair found_decl,
    clang::Expr* arg_expr) -> SemIR::ClangDeclSignatureId {
  auto signature_kind = SemIR::ClangDeclSignature::Normal;
  clang::Expr* self_expr = nullptr;
  llvm::ArrayRef<clang::Expr*> arg_exprs(arg_expr);

  // If we're performing a constructor initialization from a list, form a
  // function signature that takes a single tuple or struct pattern
  // instead of a function signature with one parameter per C++ parameter.
  if (step_kind ==
      clang::InitializationSequence::SK_ConstructorInitializationFromList) {
    // Initialization from a tuple `(a, b, c)` results in a constructor
    // function that takes a tuple pattern:
    //
    //   fn Class.Class((a: A, b: B, c: C)) -> Class;
    //
    // The source type should always be a tuple type, because we don't support
    // C++ initialization from struct types.
    auto tuple_type = context.types().TryGetAs<SemIR::TupleType>(
        context.insts().Get(source_id).type_id());
    CARBON_CHECK(tuple_type, "List initialization from non-tuple type");
    arg_exprs = cast<clang::InitListExpr>(arg_expr)->inits();
    signature_kind = SemIR::ClangDeclSignature::TuplePattern;
  }

  // In order to determine how to map the parameters, we need to build the
  // conversion sequence(s) again. Clang already threw them away. The only way
  // to do this is to "redo" overload resolution with our single candidate.
  clang::OverloadCandidateSet candidates(
      function_decl->getLocation(),
      clang::OverloadCandidateSet::CSK_InitByUserDefinedConversion);

  if (isa<clang::CXXConstructorDecl>(function_decl)) {
    // This is either tuple list initialization as described above or a
    // constructor call:
    //
    //   fn Class.Class(a: A) -> Class;
    context.clang_sema().AddOverloadCandidate(function_decl, found_decl,
                                              arg_exprs, candidates);
  } else {
    // Otherwise, the initialization is calling a conversion function
    // `Source::operator Dest`:
    //
    //   fn Source.<conversion function>(self: Source) -> Dest;
    auto* conversion_decl = cast<clang::CXXConversionDecl>(function_decl);
    self_expr = arg_expr;
    arg_exprs = {};
    context.clang_sema().AddMethodCandidate(
        conversion_decl, found_decl, conversion_decl->getParent(),
        self_expr->getType(), self_expr->Classify(context.ast_context()),
        arg_exprs, candidates);
  }

  clang::OverloadCandidateSet::iterator best;
  auto result = candidates.BestViableFunction(
      context.clang_sema(), function_decl->getLocation(), best);
  CARBON_CHECK(result == clang::OverloadingResult::OR_Success ||
               result == clang::OverloadingResult::OR_Deleted);

  return ComputeClangDeclSignatureFromBestViableFunction(
      context, best, self_expr, arg_exprs, signature_kind);
}

static auto LookupCppConversion(Context& context, SemIR::LocId loc_id,
                                SemIR::InstId source_id,
                                SemIR::TypeId dest_type_id, bool allow_explicit)
    -> SemIR::InstId {
  if (context.types().Is<SemIR::StructType>(
          context.insts().Get(source_id).type_id())) {
    // Structs can only be used to initialize C++ aggregates. That case is
    // handled by Convert, not here.
    return SemIR::InstId::None;
  }

  auto dest_type = MapToCppType(context, dest_type_id);
  if (dest_type.isNull()) {
    return SemIR::InstId::None;
  }

  auto* arg_expr = InventClangArg(context, source_id);
  // If we can't map the argument, we can't perform the conversion.
  if (!arg_expr) {
    return SemIR::InstId::None;
  }

  auto loc = GetCppLocation(context, loc_id);

  // Form a Clang initialization sequence.
  auto& sema = context.clang_sema();
  clang::InitializedEntity entity =
      clang::InitializedEntity::InitializeTemporary(dest_type);
  clang::InitializationKind kind =
      allow_explicit ? clang::InitializationKind::CreateDirect(
                           loc, /*LParenLoc=*/clang::SourceLocation(),
                           /*RParenLoc=*/clang::SourceLocation())
                     : clang::InitializationKind::CreateCopy(
                           loc, /*EqualLoc=*/clang::SourceLocation());
  clang::MultiExprArg args(arg_expr);
  // `(a, b) as T` uses `T{a, b}`, not `T({a, b})`. The latter would introduce
  // a redundant extra copy.
  // TODO: We need to communicate this back to the caller so they know to call
  // the constructor with an exploded argument list somehow.
  if (allow_explicit && isa<clang::InitListExpr>(arg_expr)) {
    kind = clang::InitializationKind::CreateDirectList(loc);
  }
  clang::InitializationSequence init(sema, entity, kind, args);

  if (init.Failed()) {
    // TODO: Are there initialization failures that we should translate into
    // errors rather than a missing conversion?
    return SemIR::InstId::None;
  }

  // Scan the steps looking for user-defined conversions. For now we just find
  // and return the first such conversion function. We skip over standard
  // conversions; we'll perform those using the Carbon rules as part of calling
  // the C++ conversion function.
  for (const auto& step : init.steps()) {
    switch (step.Kind) {
      case clang::InitializationSequence::SK_UserConversion:
      case clang::InitializationSequence::SK_ConstructorInitialization:
      case clang::InitializationSequence::SK_StdInitializerListConstructorCall:
      case clang::InitializationSequence::
          SK_ConstructorInitializationFromList: {
        if (auto* ctor =
                dyn_cast<clang::CXXConstructorDecl>(step.Function.Function);
            ctor && ctor->isCopyOrMoveConstructor()) {
          // Skip copy / move constructor calls. They shouldn't be performed
          // this way because they're not considered conversions in Carbon, and
          // will frequently lead to infinite recursion because we'll end up
          // back here when attempting to convert the argument.
          continue;
        }

        if (sema.DiagnoseUseOfOverloadedDecl(step.Function.Function, loc)) {
          return SemIR::ErrorInst::InstId;
        }

        sema.MarkFunctionReferenced(loc, step.Function.Function);

        SemIR::ClangDeclSignatureId signature_id =
            GetConversionSignatureToImport(context, source_id, step.Kind,
                                           step.Function.Function,
                                           step.Function.FoundDecl, arg_expr);
        auto result_id = ImportCppFunctionDecl(
            context, loc_id, step.Function.Function, signature_id);
        if (auto fn_decl = context.insts().TryGetAsWithId<SemIR::FunctionDecl>(
                result_id)) {
          CheckCppOverloadAccess(context, loc_id, step.Function.FoundDecl,
                                 fn_decl->inst_id);
        } else {
          CARBON_CHECK(result_id == SemIR::ErrorInst::InstId);
        }

        // TODO: There may be other conversions later in the sequence that we
        // need to model; we've only applied the first one here.
        return result_id;
      }

      case clang::InitializationSequence::SK_StdInitializerList: {
        return MakeCppStdInitializerListMake(
            context, loc_id, step.Type,
            cast<clang::InitListExpr>(arg_expr)->getNumInits());
      }

      case clang::InitializationSequence::SK_ListInitialization: {
        // Aggregate initialization is handled by the normal Carbon conversion
        // logic, so we ignore it here.
        // TODO: So far we only support aggregate initialization for arrays and
        // empty classes.
        continue;
      }

      case clang::InitializationSequence::SK_ConversionSequence:
      case clang::InitializationSequence::SK_ConversionSequenceNoNarrowing: {
        // Implicit conversions are handled by the normal Carbon conversion
        // logic, so we ignore them here.
        continue;
      }

      default: {
        // TODO: Handle other kinds of initialization steps. For now we assume
        // they will be handled by our function call logic and we can skip them.
        RawStringOstream os;
        os << "Unsupported initialization sequence:\n";
        init.dump(os);
        context.TODO(loc_id, os.TakeStr());
        return SemIR::ErrorInst::InstId;
      }
    }
  }

  return SemIR::InstId::None;
}

namespace {
// Information about a C++ overloaded operator that we might map into a Carbon
// builtin function.
struct OverloadedOperatorInfo {
  enum ReturnType { FirstArgType, Bool };

  // The name for the function used to implement this operator. This is usually
  // `Op`. This mostly only affects the mangled name, but might show up in
  // diagnostics.
  CoreIdentifier op_name = CoreIdentifier::Op;

  // The builtin function used to implement this operator. For now we're only
  // supporting enum types, so this should be an int builtin.
  SemIR::BuiltinFunctionKind builtin_kind = SemIR::BuiltinFunctionKind::None;

  // The return type to produce for the overloaded operator.
  ReturnType return_type;
};
}  // namespace

// Determine what kind of Carbon builtin function should be used to represent
// the given C++ overloaded operator.
static auto GetBuiltinOperatorInfo(clang::OverloadedOperatorKind kind)
    -> OverloadedOperatorInfo {
  using OperatorTable =
      std::array<OverloadedOperatorInfo, clang::NUM_OVERLOADED_OPERATORS>;
  static constexpr OperatorTable OpTable = [] {
    OperatorTable table = {};

    // Bitwise operators. In C++, the return type is computed with the usual
    // arithmetic conversions, but we will just use the type of the arguments.
    table[clang::OO_Amp] = {
        .builtin_kind = SemIR::BuiltinFunctionKind::IntAnd,
        .return_type = OverloadedOperatorInfo::ReturnType::FirstArgType};
    table[clang::OO_Pipe] = {
        .builtin_kind = SemIR::BuiltinFunctionKind::IntOr,
        .return_type = OverloadedOperatorInfo::ReturnType::FirstArgType};
    table[clang::OO_Caret] = {
        .builtin_kind = SemIR::BuiltinFunctionKind::IntXor,
        .return_type = OverloadedOperatorInfo::ReturnType::FirstArgType};
    table[clang::OO_Tilde] = {
        .builtin_kind = SemIR::BuiltinFunctionKind::IntComplement,
        .return_type = OverloadedOperatorInfo::ReturnType::FirstArgType};

    // Comparison operators.
    table[clang::OO_EqualEqual] = {
        .op_name = CoreIdentifier::Equal,
        .builtin_kind = SemIR::BuiltinFunctionKind::IntEq,
        .return_type = OverloadedOperatorInfo::ReturnType::Bool};
    table[clang::OO_ExclaimEqual] = {
        .op_name = CoreIdentifier::NotEqual,
        .builtin_kind = SemIR::BuiltinFunctionKind::IntNeq,
        .return_type = OverloadedOperatorInfo::ReturnType::Bool};
    table[clang::OO_Less] = {
        .op_name = CoreIdentifier::Less,
        .builtin_kind = SemIR::BuiltinFunctionKind::IntLess,
        .return_type = OverloadedOperatorInfo::ReturnType::Bool};
    table[clang::OO_LessEqual] = {
        .op_name = CoreIdentifier::LessOrEquivalent,
        .builtin_kind = SemIR::BuiltinFunctionKind::IntLessEq,
        .return_type = OverloadedOperatorInfo::ReturnType::Bool};
    table[clang::OO_Greater] = {
        .op_name = CoreIdentifier::Greater,
        .builtin_kind = SemIR::BuiltinFunctionKind::IntGreater,
        .return_type = OverloadedOperatorInfo::ReturnType::Bool};
    table[clang::OO_GreaterEqual] = {
        .op_name = CoreIdentifier::GreaterOrEquivalent,
        .builtin_kind = SemIR::BuiltinFunctionKind::IntGreaterEq,
        .return_type = OverloadedOperatorInfo::ReturnType::Bool};

    return table;
  }();
  return OpTable[kind];
}

// Builds a Carbon builtin function declaration corresponding to an overload
// candidate that selected a C++ builtin operator. Returns None if no
// corresponding builtin function could or should be built.
static auto TryBuildBuiltinOperator(
    Context& context, SemIR::LocId loc_id,
    clang::OverloadedOperatorKind op_kind,
    clang::OverloadCandidateSet::iterator candidate) -> SemIR::InstId {
  auto info = GetBuiltinOperatorInfo(op_kind);
  if (info.builtin_kind == SemIR::BuiltinFunctionKind::None) {
    return SemIR::InstId::None;
  }

  // Import the argument types. For now, we only accept enum types.
  // TODO: Consider expanding this to other types.
  llvm::SmallVector<SemIR::TypeId, 2> arg_type_ids;
  for (const auto& conversion : candidate->Conversions) {
    // Get the type of the argument that overload resolution wanted to pass to
    // the overload candidate, after any user-defined implicit conversions but
    // before the final standard conversion sequence, to find an enum type prior
    // to promotion.
    clang::QualType converted_type;
    if (conversion.isStandard()) {
      converted_type = conversion.Standard.getFromType();
    } else if (conversion.isUserDefined()) {
      converted_type = conversion.UserDefined.After.getFromType();
    } else {
      // Unexpected kind of conversion sequence.
      return SemIR::InstId::None;
    }
    if (!converted_type->isEnumeralType()) {
      return SemIR::InstId::None;
    }
    auto arg_type_id = ImportCppType(context, loc_id, converted_type).type_id;
    if (!arg_type_id.has_value() || arg_type_id == SemIR::ErrorInst::TypeId) {
      return SemIR::InstId::None;
    }
    arg_type_ids.push_back(arg_type_id);
  }
  CARBON_CHECK(arg_type_ids.size() == 1 || arg_type_ids.size() == 2);

  // For now we only accept homogeneous operators.
  if (arg_type_ids.size() == 2 && arg_type_ids[0] != arg_type_ids[1]) {
    return SemIR::InstId::None;
  }

  // Compute the return type.
  auto return_type_id = SemIR::TypeId::None;
  switch (info.return_type) {
    case OverloadedOperatorInfo::FirstArgType:
      return_type_id = arg_type_ids[0];
      break;
    case OverloadedOperatorInfo::Bool:
      return_type_id =
          context.types().GetTypeIdForTypeInstId(SemIR::BoolType::TypeInstId);
      break;
  }

  return MakeBuiltinOperatorFunction(context, arg_type_ids, return_type_id,
                                     info.op_name, info.builtin_kind);
}

namespace {
struct DiagnoseIncompleteOperandTypeInCppOperatorLookup {
  Context& context;
  SemIR::TypeId arg_type_id;
  SemIR::LocId loc_id;

  void operator()(auto& builder) const {
    CARBON_DIAGNOSTIC(
        IncompleteOperandTypeInCppOperatorLookup, Context,
        "looking up a C++ operator with incomplete operand type {0}",
        SemIR::TypeId);
    builder.Context(loc_id, IncompleteOperandTypeInCppOperatorLookup,
                    arg_type_id);
  }
};
}  // namespace

static auto FindClangOperator(Context& context, SemIR::LocId loc_id,
                              clang::OverloadedOperatorKind op_kind,
                              llvm::ArrayRef<clang::Expr*> arg_exprs)
    -> SemIR::InstId;

auto LookupCppOperator(Context& context, SemIR::LocId loc_id, Operator op,
                       llvm::ArrayRef<SemIR::TypeId> arg_type_ids)
    -> SemIR::InstId {
  // Register an annotation scope to flush any Clang diagnostics when we return.
  // This is important to ensure that Clang diagnostics are properly interleaved
  // with Carbon diagnostics.
  Diagnostics::AnnotationScope annotate_diagnostics(&context.emitter(),
                                                    [](auto& /*builder*/) {});

  if (op.interface_name == CoreIdentifier::ImplicitAs ||
      op.interface_name == CoreIdentifier::As) {
    context.TODO(loc_id, "handle `as` operator when passed a type");
    return SemIR::ErrorInst::InstId;
  }

  auto op_kind =
      GetClangOperatorKind(context, loc_id, op.interface_name, op.op_name);
  if (!op_kind) {
    return SemIR::ErrorInst::InstId;
  }

  for (SemIR::TypeId arg_type_id : arg_type_ids) {
    if (!RequireCompleteType(context, arg_type_id, loc_id,
                             DiagnoseIncompleteOperandTypeInCppOperatorLookup{
                                 .context = context,
                                 .arg_type_id = arg_type_id,
                                 .loc_id = loc_id})) {
      return SemIR::ErrorInst::InstId;
    }
  }

  struct Operand {
    using enum clang::ExprValueKind;
    explicit Operand(clang::QualType type)
        : type(type),
          expression({}, type,
                     type->isLValueReferenceType()   ? VK_LValue
                     : type->isRValueReferenceType() ? VK_XValue
                                                     : VK_PRValue) {}
    clang::QualType type;
    clang::OpaqueValueExpr expression;
  };

  auto cpp_type = MapToCppType(context, arg_type_ids[0]);
  if (cpp_type.isNull()) {
    return SemIR::InstId::None;
  }
  auto arg0 = Operand(cpp_type);
  if (arg_type_ids.size() == 1) {
    return FindClangOperator(context, loc_id, *op_kind, {&arg0.expression});
  }
  CARBON_CHECK(arg_type_ids.size() == 2);
  cpp_type = MapToCppType(context, arg_type_ids[1]);
  if (cpp_type.isNull()) {
    return SemIR::InstId::None;
  }
  auto arg1 = Operand(cpp_type);
  return FindClangOperator(context, loc_id, *op_kind,
                           {&arg0.expression, &arg1.expression});
}

auto LookupCppOperator(Context& context, SemIR::LocId loc_id, Operator op,
                       llvm::ArrayRef<SemIR::InstId> arg_ids) -> SemIR::InstId {
  // Register an annotation scope to flush any Clang diagnostics when we return.
  // This is important to ensure that Clang diagnostics are properly interleaved
  // with Carbon diagnostics.
  Diagnostics::AnnotationScope annotate_diagnostics(&context.emitter(),
                                                    [](auto& /*builder*/) {});

  // We can only handle concrete types in LookupCppOperator.
  for (auto arg_id : arg_ids) {
    auto type_id = context.insts().Get(arg_id).type_id();
    if (type_id.is_symbolic()) {
      return SemIR::InstId::None;
    }
  }

  // Handle `ImplicitAs` and `As`.
  if (op.interface_name == CoreIdentifier::ImplicitAs ||
      op.interface_name == CoreIdentifier::As) {
    if (op.interface_args_ref.size() != 1 || arg_ids.size() != 1) {
      return SemIR::InstId::None;
    }
    // The argument is the destination type for both interfaces.
    auto dest_const_id =
        context.constant_values().Get(op.interface_args_ref[0]);
    auto dest_type_id =
        context.types().TryGetTypeIdForTypeConstantId(dest_const_id);
    if (!dest_type_id.has_value()) {
      return SemIR::InstId::None;
    }

    return LookupCppConversion(
        context, loc_id, arg_ids[0], dest_type_id,
        /*allow_explicit=*/op.interface_name == CoreIdentifier::As);
  }

  auto op_kind =
      GetClangOperatorKind(context, loc_id, op.interface_name, op.op_name);
  if (!op_kind) {
    return SemIR::InstId::None;
  }

  // Make sure all operands are complete before lookup.
  for (SemIR::InstId arg_id : arg_ids) {
    SemIR::TypeId arg_type_id = context.insts().Get(arg_id).type_id();
    if (!RequireCompleteType(context, arg_type_id, loc_id,
                             DiagnoseIncompleteOperandTypeInCppOperatorLookup{
                                 .context = context,
                                 .arg_type_id = arg_type_id,
                                 .loc_id = loc_id})) {
      return SemIR::ErrorInst::InstId;
    }
  }

  auto maybe_arg_exprs = InventClangArgs(context, arg_ids);
  if (!maybe_arg_exprs.has_value()) {
    return SemIR::ErrorInst::InstId;
  }

  return FindClangOperator(context, loc_id, *op_kind, *maybe_arg_exprs);
}

static auto FindClangOperator(Context& context, SemIR::LocId loc_id,
                              clang::OverloadedOperatorKind op_kind,
                              llvm::ArrayRef<clang::Expr*> arg_exprs)
    -> SemIR::InstId {
  clang::SourceLocation loc = GetCppLocation(context, loc_id);
  clang::OverloadCandidateSet::OperatorRewriteInfo operator_rewrite_info(
      op_kind, loc, /*AllowRewritten=*/true);
  clang::OverloadCandidateSet candidate_set(
      loc, clang::OverloadCandidateSet::CSK_Operator, operator_rewrite_info);

  clang::Sema& sema = context.clang_sema();

  // This works for both unary and binary operators.
  sema.LookupOverloadedBinOp(candidate_set, op_kind, clang::UnresolvedSet<0>{},
                             arg_exprs);

  clang::OverloadCandidateSet::iterator best_viable_fn;
  switch (candidate_set.BestViableFunction(sema, loc, best_viable_fn)) {
    case clang::OverloadingResult::OR_Success: {
      if (!best_viable_fn->Function) {
        // The best viable candidate was a builtin. Let the Carbon operator
        // machinery handle that.
        CARBON_CHECK(!best_viable_fn->RewriteKind,
                     "Rewrite targeted builtin operator");
        return TryBuildBuiltinOperator(context, loc_id, op_kind,
                                       best_viable_fn);
      }
      if (best_viable_fn->RewriteKind) {
        context.TODO(
            loc_id,
            llvm::formatv("Rewriting operator{0} using {1} is not supported",
                          clang::getOperatorSpelling(
                              candidate_set.getRewriteInfo().OriginalOperator),
                          best_viable_fn->Function->getNameAsString()));
        return SemIR::ErrorInst::InstId;
      }
      sema.MarkFunctionReferenced(loc, best_viable_fn->Function);

      // If this is an operator method, the first arg will be used as self.
      clang::Expr* self_expr = nullptr;
      auto arg_exprs_for_signature = arg_exprs;
      if (IsObjectMemberFunction(*best_viable_fn->Function)) {
        self_expr = arg_exprs_for_signature.consume_front();
      }

      SemIR::ClangDeclSignatureId signature_id =
          ComputeClangDeclSignatureFromBestViableFunction(
              context, best_viable_fn, self_expr, arg_exprs_for_signature);

      auto result_id = ImportCppFunctionDecl(
          context, loc_id, best_viable_fn->Function, signature_id);
      if (result_id != SemIR::ErrorInst::InstId) {
        CheckCppOverloadAccess(
            context, loc_id, best_viable_fn->FoundDecl,
            context.insts().GetAsKnownInstId<SemIR::FunctionDecl>(result_id));
      }
      return result_id;
    }
    case clang::OverloadingResult::OR_No_Viable_Function: {
      // OK, didn't find a viable C++ candidate, but this is not an error, as
      // there might be a Carbon candidate.
      return SemIR::InstId::None;
    }
    case clang::OverloadingResult::OR_Ambiguous: {
      const char* spelling = clang::getOperatorSpelling(op_kind);
      candidate_set.NoteCandidates(
          clang::PartialDiagnosticAt(
              loc, sema.PDiag(clang::diag::err_ovl_ambiguous_oper_binary)
                       << spelling << arg_exprs[0]->getType()
                       << arg_exprs[1]->getType()),
          sema, clang::OCD_AmbiguousCandidates, arg_exprs, spelling, loc);
      return SemIR::ErrorInst::InstId;
    }
    case clang::OverloadingResult::OR_Deleted:
      const char* spelling = clang::getOperatorSpelling(op_kind);
      auto* message = best_viable_fn->Function->getDeletedMessage();
      // The best viable function might be a different operator if the best
      // candidate is a rewritten candidate, so use the operator kind of the
      // candidate itself in the diagnostic.
      candidate_set.NoteCandidates(
          clang::PartialDiagnosticAt(
              loc, sema.PDiag(clang::diag::err_ovl_deleted_oper)
                       << clang::getOperatorSpelling(
                              best_viable_fn->Function->getOverloadedOperator())
                       << (message != nullptr)
                       << (message ? message->getString() : llvm::StringRef())),
          sema, clang::OCD_AllCandidates, arg_exprs, spelling, loc);
      return SemIR::ErrorInst::InstId;
  }
}

auto IsCppOperatorMethodDecl(clang::Decl* decl) -> bool {
  auto* clang_method_decl = dyn_cast<clang::CXXMethodDecl>(decl);
  return clang_method_decl &&
         (clang_method_decl->isOverloadedOperator() ||
          isa<clang::CXXConversionDecl>(clang_method_decl));
}

static auto GetAsCppFunctionDecl(Context& context, SemIR::InstId inst_id)
    -> clang::FunctionDecl* {
  if (inst_id == SemIR::InstId::None) {
    return nullptr;
  }
  auto function_type = context.types().TryGetAs<SemIR::FunctionType>(
      context.insts().Get(inst_id).type_id());
  if (!function_type) {
    return nullptr;
  }
  const auto* clang_decl = context.clang_decls().Lookup(
      context.functions().Get(function_type->function_id).first_decl_id());
  return clang_decl ? dyn_cast<clang::FunctionDecl>(clang_decl->decl())
                    : nullptr;
}

auto IsCppConstructorOrNonMethod(Context& context, SemIR::InstId inst_id)
    -> bool {
  auto* function_decl = GetAsCppFunctionDecl(context, inst_id);
  if (!function_decl) {
    return false;
  }
  if (isa<clang::CXXConstructorDecl>(function_decl)) {
    return true;
  }
  return !isa<clang::CXXMethodDecl>(function_decl);
}

}  // namespace Carbon::Check
