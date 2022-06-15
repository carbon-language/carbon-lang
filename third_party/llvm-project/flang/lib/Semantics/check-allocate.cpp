//===-- lib/Semantics/check-allocate.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-allocate.h"
#include "assignment.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

namespace Fortran::semantics {

struct AllocateCheckerInfo {
  const DeclTypeSpec *typeSpec{nullptr};
  std::optional<evaluate::DynamicType> sourceExprType;
  std::optional<parser::CharBlock> sourceExprLoc;
  std::optional<parser::CharBlock> typeSpecLoc;
  int sourceExprRank{0}; // only valid if gotMold || gotSource
  bool gotStat{false};
  bool gotMsg{false};
  bool gotTypeSpec{false};
  bool gotSource{false};
  bool gotMold{false};
};

class AllocationCheckerHelper {
public:
  AllocationCheckerHelper(
      const parser::Allocation &alloc, AllocateCheckerInfo &info)
      : allocateInfo_{info}, allocateObject_{std::get<parser::AllocateObject>(
                                 alloc.t)},
        name_{parser::GetLastName(allocateObject_)},
        symbol_{name_.symbol ? &name_.symbol->GetUltimate() : nullptr},
        type_{symbol_ ? symbol_->GetType() : nullptr},
        allocateShapeSpecRank_{ShapeSpecRank(alloc)}, rank_{symbol_
                                                              ? symbol_->Rank()
                                                              : 0},
        allocateCoarraySpecRank_{CoarraySpecRank(alloc)},
        corank_{symbol_ ? symbol_->Corank() : 0} {}

  bool RunChecks(SemanticsContext &context);

private:
  bool hasAllocateShapeSpecList() const { return allocateShapeSpecRank_ != 0; }
  bool hasAllocateCoarraySpec() const { return allocateCoarraySpecRank_ != 0; }
  bool RunCoarrayRelatedChecks(SemanticsContext &) const;

  static int ShapeSpecRank(const parser::Allocation &allocation) {
    return static_cast<int>(
        std::get<std::list<parser::AllocateShapeSpec>>(allocation.t).size());
  }

  static int CoarraySpecRank(const parser::Allocation &allocation) {
    if (const auto &coarraySpec{
            std::get<std::optional<parser::AllocateCoarraySpec>>(
                allocation.t)}) {
      return std::get<std::list<parser::AllocateCoshapeSpec>>(coarraySpec->t)
                 .size() +
          1;
    } else {
      return 0;
    }
  }

  void GatherAllocationBasicInfo() {
    if (type_->category() == DeclTypeSpec::Category::Character) {
      hasDeferredTypeParameter_ =
          type_->characterTypeSpec().length().isDeferred();
    } else if (const DerivedTypeSpec * derivedTypeSpec{type_->AsDerived()}) {
      for (const auto &pair : derivedTypeSpec->parameters()) {
        hasDeferredTypeParameter_ |= pair.second.isDeferred();
      }
      isAbstract_ = derivedTypeSpec->typeSymbol().attrs().test(Attr::ABSTRACT);
    }
    isUnlimitedPolymorphic_ =
        type_->category() == DeclTypeSpec::Category::ClassStar;
  }

  AllocateCheckerInfo &allocateInfo_;
  const parser::AllocateObject &allocateObject_;
  const parser::Name &name_;
  const Symbol *symbol_{nullptr};
  const DeclTypeSpec *type_{nullptr};
  const int allocateShapeSpecRank_;
  const int rank_{0};
  const int allocateCoarraySpecRank_;
  const int corank_{0};
  bool hasDeferredTypeParameter_{false};
  bool isUnlimitedPolymorphic_{false};
  bool isAbstract_{false};
};

static std::optional<AllocateCheckerInfo> CheckAllocateOptions(
    const parser::AllocateStmt &allocateStmt, SemanticsContext &context) {
  AllocateCheckerInfo info;
  bool stopCheckingAllocate{false}; // for errors that would lead to ambiguity
  if (const auto &typeSpec{
          std::get<std::optional<parser::TypeSpec>>(allocateStmt.t)}) {
    info.typeSpec = typeSpec->declTypeSpec;
    if (!info.typeSpec) {
      CHECK(context.AnyFatalError());
      return std::nullopt;
    }
    info.gotTypeSpec = true;
    info.typeSpecLoc = parser::FindSourceLocation(*typeSpec);
    if (const DerivedTypeSpec * derived{info.typeSpec->AsDerived()}) {
      // C937
      if (auto it{FindCoarrayUltimateComponent(*derived)}) {
        context
            .Say("Type-spec in ALLOCATE must not specify a type with a coarray"
                 " ultimate component"_err_en_US)
            .Attach(it->name(),
                "Type '%s' has coarray ultimate component '%s' declared here"_en_US,
                info.typeSpec->AsFortran(), it.BuildResultDesignatorName());
      }
    }
  }

  const parser::Expr *parserSourceExpr{nullptr};
  for (const parser::AllocOpt &allocOpt :
      std::get<std::list<parser::AllocOpt>>(allocateStmt.t)) {
    common::visit(
        common::visitors{
            [&](const parser::StatOrErrmsg &statOrErr) {
              common::visit(
                  common::visitors{
                      [&](const parser::StatVariable &) {
                        if (info.gotStat) { // C943
                          context.Say(
                              "STAT may not be duplicated in a ALLOCATE statement"_err_en_US);
                        }
                        info.gotStat = true;
                      },
                      [&](const parser::MsgVariable &) {
                        if (info.gotMsg) { // C943
                          context.Say(
                              "ERRMSG may not be duplicated in a ALLOCATE statement"_err_en_US);
                        }
                        info.gotMsg = true;
                      },
                  },
                  statOrErr.u);
            },
            [&](const parser::AllocOpt::Source &source) {
              if (info.gotSource) { // C943
                context.Say(
                    "SOURCE may not be duplicated in a ALLOCATE statement"_err_en_US);
                stopCheckingAllocate = true;
              }
              if (info.gotMold || info.gotTypeSpec) { // C944
                context.Say(
                    "At most one of source-expr and type-spec may appear in a ALLOCATE statement"_err_en_US);
                stopCheckingAllocate = true;
              }
              parserSourceExpr = &source.v.value();
              info.gotSource = true;
            },
            [&](const parser::AllocOpt::Mold &mold) {
              if (info.gotMold) { // C943
                context.Say(
                    "MOLD may not be duplicated in a ALLOCATE statement"_err_en_US);
                stopCheckingAllocate = true;
              }
              if (info.gotSource || info.gotTypeSpec) { // C944
                context.Say(
                    "At most one of source-expr and type-spec may appear in a ALLOCATE statement"_err_en_US);
                stopCheckingAllocate = true;
              }
              parserSourceExpr = &mold.v.value();
              info.gotMold = true;
            },
        },
        allocOpt.u);
  }

  if (stopCheckingAllocate) {
    return std::nullopt;
  }

  if (info.gotSource || info.gotMold) {
    if (const auto *expr{GetExpr(context, DEREF(parserSourceExpr))}) {
      parser::CharBlock at{parserSourceExpr->source};
      info.sourceExprType = expr->GetType();
      if (!info.sourceExprType) {
        context.Say(at,
            "Typeless item not allowed as SOURCE or MOLD in ALLOCATE"_err_en_US);
        return std::nullopt;
      }
      info.sourceExprRank = expr->Rank();
      info.sourceExprLoc = parserSourceExpr->source;
      if (const DerivedTypeSpec *
          derived{evaluate::GetDerivedTypeSpec(info.sourceExprType)}) {
        // C949
        if (auto it{FindCoarrayUltimateComponent(*derived)}) {
          context
              .Say(at,
                  "SOURCE or MOLD expression must not have a type with a coarray ultimate component"_err_en_US)
              .Attach(it->name(),
                  "Type '%s' has coarray ultimate component '%s' declared here"_en_US,
                  info.sourceExprType.value().AsFortran(),
                  it.BuildResultDesignatorName());
        }
        if (info.gotSource) {
          // C948
          if (IsEventTypeOrLockType(derived)) {
            context.Say(at,
                "SOURCE expression type must not be EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV"_err_en_US);
          } else if (auto it{FindEventOrLockPotentialComponent(*derived)}) {
            context
                .Say(at,
                    "SOURCE expression type must not have potential subobject "
                    "component"
                    " of type EVENT_TYPE or LOCK_TYPE from ISO_FORTRAN_ENV"_err_en_US)
                .Attach(it->name(),
                    "Type '%s' has potential ultimate component '%s' declared here"_en_US,
                    info.sourceExprType.value().AsFortran(),
                    it.BuildResultDesignatorName());
          }
        }
      }
      if (info.gotSource) { // C1594(6) - SOURCE= restrictions when pure
        const Scope &scope{context.FindScope(at)};
        if (FindPureProcedureContaining(scope)) {
          parser::ContextualMessages messages{at, &context.messages()};
          CheckCopyabilityInPureScope(messages, *expr, scope);
        }
      }
    } else {
      // Error already reported on source expression.
      // Do not continue allocate checks.
      return std::nullopt;
    }
  }

  return info;
}

// Beware, type compatibility is not symmetric, IsTypeCompatible checks that
// type1 is type compatible with type2. Note: type parameters are not considered
// in this test.
static bool IsTypeCompatible(
    const DeclTypeSpec &type1, const DerivedTypeSpec &derivedType2) {
  if (const DerivedTypeSpec * derivedType1{type1.AsDerived()}) {
    if (type1.category() == DeclTypeSpec::Category::TypeDerived) {
      return &derivedType1->typeSymbol() == &derivedType2.typeSymbol();
    } else if (type1.category() == DeclTypeSpec::Category::ClassDerived) {
      for (const DerivedTypeSpec *parent{&derivedType2}; parent;
           parent = parent->typeSymbol().GetParentTypeSpec()) {
        if (&derivedType1->typeSymbol() == &parent->typeSymbol()) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool IsTypeCompatible(
    const DeclTypeSpec &type1, const DeclTypeSpec &type2) {
  if (type1.category() == DeclTypeSpec::Category::ClassStar) {
    // TypeStar does not make sense in allocate context because assumed type
    // cannot be allocatable (C709)
    return true;
  }
  if (const IntrinsicTypeSpec * intrinsicType2{type2.AsIntrinsic()}) {
    if (const IntrinsicTypeSpec * intrinsicType1{type1.AsIntrinsic()}) {
      return intrinsicType1->category() == intrinsicType2->category();
    } else {
      return false;
    }
  } else if (const DerivedTypeSpec * derivedType2{type2.AsDerived()}) {
    return IsTypeCompatible(type1, *derivedType2);
  }
  return false;
}

static bool IsTypeCompatible(
    const DeclTypeSpec &type1, const evaluate::DynamicType &type2) {
  if (type1.category() == DeclTypeSpec::Category::ClassStar) {
    // TypeStar does not make sense in allocate context because assumed type
    // cannot be allocatable (C709)
    return true;
  }
  if (type2.category() != evaluate::TypeCategory::Derived) {
    if (const IntrinsicTypeSpec * intrinsicType1{type1.AsIntrinsic()}) {
      return intrinsicType1->category() == type2.category();
    } else {
      return false;
    }
  } else if (!type2.IsUnlimitedPolymorphic()) {
    return IsTypeCompatible(type1, type2.GetDerivedTypeSpec());
  }
  return false;
}

// Note: Check assumes  type1 is compatible with type2. type2 may have more type
// parameters than type1 but if a type2 type parameter is assumed, then this
// check enforce that type1 has it. type1 can be unlimited polymorphic, but not
// type2.
static bool HaveSameAssumedTypeParameters(
    const DeclTypeSpec &type1, const DeclTypeSpec &type2) {
  if (type2.category() == DeclTypeSpec::Category::Character) {
    bool type2LengthIsAssumed{type2.characterTypeSpec().length().isAssumed()};
    if (type1.category() == DeclTypeSpec::Category::Character) {
      return type1.characterTypeSpec().length().isAssumed() ==
          type2LengthIsAssumed;
    }
    // It is possible to reach this if type1 is unlimited polymorphic
    return !type2LengthIsAssumed;
  } else if (const DerivedTypeSpec * derivedType2{type2.AsDerived()}) {
    int type2AssumedParametersCount{0};
    int type1AssumedParametersCount{0};
    for (const auto &pair : derivedType2->parameters()) {
      type2AssumedParametersCount += pair.second.isAssumed();
    }
    // type1 may be unlimited polymorphic
    if (const DerivedTypeSpec * derivedType1{type1.AsDerived()}) {
      for (auto it{derivedType1->parameters().begin()};
           it != derivedType1->parameters().end(); ++it) {
        if (it->second.isAssumed()) {
          ++type1AssumedParametersCount;
          const ParamValue *param{derivedType2->FindParameter(it->first)};
          if (!param || !param->isAssumed()) {
            // type1 has an assumed parameter that is not a type parameter of
            // type2 or not assumed in type2.
            return false;
          }
        }
      }
    }
    // Will return false if type2 has type parameters that are not assumed in
    // type1 or do not exist in type1
    return type1AssumedParametersCount == type2AssumedParametersCount;
  }
  return true; // other intrinsic types have no length type parameters
}

static std::optional<std::int64_t> GetTypeParameterInt64Value(
    const Symbol &parameterSymbol, const DerivedTypeSpec &derivedType) {
  if (const ParamValue *
      paramValue{derivedType.FindParameter(parameterSymbol.name())}) {
    return evaluate::ToInt64(paramValue->GetExplicit());
  } else {
    return std::nullopt;
  }
}

// HaveCompatibleKindParameters functions assume type1 is type compatible with
// type2 (except for kind type parameters)
static bool HaveCompatibleKindParameters(
    const DerivedTypeSpec &derivedType1, const DerivedTypeSpec &derivedType2) {
  for (const Symbol &symbol :
      OrderParameterDeclarations(derivedType1.typeSymbol())) {
    if (symbol.get<TypeParamDetails>().attr() == common::TypeParamAttr::Kind) {
      // At this point, it should have been ensured that these contain integer
      // constants, so die if this is not the case.
      if (GetTypeParameterInt64Value(symbol, derivedType1).value() !=
          GetTypeParameterInt64Value(symbol, derivedType2).value()) {
        return false;
      }
    }
  }
  return true;
}

static bool HaveCompatibleKindParameters(
    const DeclTypeSpec &type1, const evaluate::DynamicType &type2) {
  if (type1.category() == DeclTypeSpec::Category::ClassStar) {
    return true;
  }
  if (const IntrinsicTypeSpec * intrinsicType1{type1.AsIntrinsic()}) {
    return evaluate::ToInt64(intrinsicType1->kind()).value() == type2.kind();
  } else if (type2.IsUnlimitedPolymorphic()) {
    return false;
  } else if (const DerivedTypeSpec * derivedType1{type1.AsDerived()}) {
    return HaveCompatibleKindParameters(
        *derivedType1, type2.GetDerivedTypeSpec());
  } else {
    common::die("unexpected type1 category");
  }
}

static bool HaveCompatibleKindParameters(
    const DeclTypeSpec &type1, const DeclTypeSpec &type2) {
  if (type1.category() == DeclTypeSpec::Category::ClassStar) {
    return true;
  }
  if (const IntrinsicTypeSpec * intrinsicType1{type1.AsIntrinsic()}) {
    return intrinsicType1->kind() == DEREF(type2.AsIntrinsic()).kind();
  } else if (const DerivedTypeSpec * derivedType1{type1.AsDerived()}) {
    return HaveCompatibleKindParameters(
        *derivedType1, DEREF(type2.AsDerived()));
  } else {
    common::die("unexpected type1 category");
  }
}

bool AllocationCheckerHelper::RunChecks(SemanticsContext &context) {
  if (!symbol_) {
    CHECK(context.AnyFatalError());
    return false;
  }
  if (!IsVariableName(*symbol_)) { // C932 pre-requisite
    context.Say(name_.source,
        "Name in ALLOCATE statement must be a variable name"_err_en_US);
    return false;
  }
  if (!type_) {
    // This is done after variable check because a user could have put
    // a subroutine name in allocate for instance which is a symbol with
    // no type.
    CHECK(context.AnyFatalError());
    return false;
  }
  GatherAllocationBasicInfo();
  if (!IsAllocatableOrPointer(*symbol_)) { // C932
    context.Say(name_.source,
        "Entity in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
    return false;
  }
  bool gotSourceExprOrTypeSpec{allocateInfo_.gotMold ||
      allocateInfo_.gotTypeSpec || allocateInfo_.gotSource};
  if (hasDeferredTypeParameter_ && !gotSourceExprOrTypeSpec) {
    // C933
    context.Say(name_.source,
        "Either type-spec or source-expr must appear in ALLOCATE when allocatable object has a deferred type parameters"_err_en_US);
    return false;
  }
  if (isUnlimitedPolymorphic_ && !gotSourceExprOrTypeSpec) {
    // C933
    context.Say(name_.source,
        "Either type-spec or source-expr must appear in ALLOCATE when allocatable object is unlimited polymorphic"_err_en_US);
    return false;
  }
  if (isAbstract_ && !gotSourceExprOrTypeSpec) {
    // C933
    context.Say(name_.source,
        "Either type-spec or source-expr must appear in ALLOCATE when allocatable object is of abstract type"_err_en_US);
    return false;
  }
  if (allocateInfo_.gotTypeSpec) {
    if (!IsTypeCompatible(*type_, *allocateInfo_.typeSpec)) {
      // C934
      context.Say(name_.source,
          "Allocatable object in ALLOCATE must be type compatible with type-spec"_err_en_US);
      return false;
    }
    if (!HaveCompatibleKindParameters(*type_, *allocateInfo_.typeSpec)) {
      context.Say(name_.source,
          // C936
          "Kind type parameters of allocatable object in ALLOCATE must be the same as the corresponding ones in type-spec"_err_en_US);
      return false;
    }
    if (!HaveSameAssumedTypeParameters(*type_, *allocateInfo_.typeSpec)) {
      // C935
      context.Say(name_.source,
          "Type parameters in type-spec must be assumed if and only if they are assumed for allocatable object in ALLOCATE"_err_en_US);
      return false;
    }
  } else if (allocateInfo_.gotSource || allocateInfo_.gotMold) {
    if (!IsTypeCompatible(*type_, allocateInfo_.sourceExprType.value())) {
      // first part of C945
      context.Say(name_.source,
          "Allocatable object in ALLOCATE must be type compatible with source expression from MOLD or SOURCE"_err_en_US);
      return false;
    }
    if (!HaveCompatibleKindParameters(
            *type_, allocateInfo_.sourceExprType.value())) {
      // C946
      context.Say(name_.source,
          "Kind type parameters of allocatable object must be the same as the corresponding ones of SOURCE or MOLD expression"_err_en_US);
      return false;
    }
  }
  // Shape related checks
  if (rank_ > 0) {
    if (!hasAllocateShapeSpecList()) {
      // C939
      if (!(allocateInfo_.gotSource || allocateInfo_.gotMold)) {
        context.Say(name_.source,
            "Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD"_err_en_US);
        return false;
      } else {
        if (allocateInfo_.sourceExprRank != rank_) {
          context
              .Say(name_.source,
                  "Arrays in ALLOCATE must have a shape specification or an expression of the same rank must appear in SOURCE or MOLD"_err_en_US)
              .Attach(allocateInfo_.sourceExprLoc.value(),
                  "Expression in %s has rank %d but allocatable object has rank %d"_en_US,
                  allocateInfo_.gotSource ? "SOURCE" : "MOLD",
                  allocateInfo_.sourceExprRank, rank_);
          return false;
        }
      }
    } else {
      // first part of C942
      if (allocateShapeSpecRank_ != rank_) {
        context
            .Say(name_.source,
                "The number of shape specifications, when they appear, must match the rank of allocatable object"_err_en_US)
            .Attach(symbol_->name(), "Declared here with rank %d"_en_US, rank_);
        return false;
      }
    }
  } else {
    // C940
    if (hasAllocateShapeSpecList()) {
      context.Say(name_.source,
          "Shape specifications must not appear when allocatable object is scalar"_err_en_US);
      return false;
    }
  }
  // second and last part of C945
  if (allocateInfo_.gotSource && allocateInfo_.sourceExprRank &&
      allocateInfo_.sourceExprRank != rank_) {
    context
        .Say(name_.source,
            "If SOURCE appears, the related expression must be scalar or have the same rank as each allocatable object in ALLOCATE"_err_en_US)
        .Attach(allocateInfo_.sourceExprLoc.value(),
            "SOURCE expression has rank %d"_en_US, allocateInfo_.sourceExprRank)
        .Attach(symbol_->name(),
            "Allocatable object declared here with rank %d"_en_US, rank_);
    return false;
  }
  context.CheckIndexVarRedefine(name_);
  return RunCoarrayRelatedChecks(context);
}

bool AllocationCheckerHelper::RunCoarrayRelatedChecks(
    SemanticsContext &context) const {
  if (!symbol_) {
    CHECK(context.AnyFatalError());
    return false;
  }
  if (evaluate::IsCoarray(*symbol_)) {
    if (allocateInfo_.gotTypeSpec) {
      // C938
      if (const DerivedTypeSpec *
          derived{allocateInfo_.typeSpec->AsDerived()}) {
        if (IsTeamType(derived)) {
          context
              .Say(allocateInfo_.typeSpecLoc.value(),
                  "Type-Spec in ALLOCATE must not be TEAM_TYPE from ISO_FORTRAN_ENV when an allocatable object is a coarray"_err_en_US)
              .Attach(name_.source, "'%s' is a coarray"_en_US, name_.source);
          return false;
        } else if (IsIsoCType(derived)) {
          context
              .Say(allocateInfo_.typeSpecLoc.value(),
                  "Type-Spec in ALLOCATE must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray"_err_en_US)
              .Attach(name_.source, "'%s' is a coarray"_en_US, name_.source);
          return false;
        }
      }
    } else if (allocateInfo_.gotSource || allocateInfo_.gotMold) {
      // C948
      const evaluate::DynamicType &sourceType{
          allocateInfo_.sourceExprType.value()};
      if (const auto *derived{evaluate::GetDerivedTypeSpec(sourceType)}) {
        if (IsTeamType(derived)) {
          context
              .Say(allocateInfo_.sourceExprLoc.value(),
                  "SOURCE or MOLD expression type must not be TEAM_TYPE from ISO_FORTRAN_ENV when an allocatable object is a coarray"_err_en_US)
              .Attach(name_.source, "'%s' is a coarray"_en_US, name_.source);
          return false;
        } else if (IsIsoCType(derived)) {
          context
              .Say(allocateInfo_.sourceExprLoc.value(),
                  "SOURCE or MOLD expression type must not be C_PTR or C_FUNPTR from ISO_C_BINDING when an allocatable object is a coarray"_err_en_US)
              .Attach(name_.source, "'%s' is a coarray"_en_US, name_.source);
          return false;
        }
      }
    }
    if (!hasAllocateCoarraySpec()) {
      // C941
      context.Say(name_.source,
          "Coarray specification must appear in ALLOCATE when allocatable object is a coarray"_err_en_US);
      return false;
    } else {
      if (allocateCoarraySpecRank_ != corank_) {
        // Second and last part of C942
        context
            .Say(name_.source,
                "Corank of coarray specification in ALLOCATE must match corank of alloctable coarray"_err_en_US)
            .Attach(
                symbol_->name(), "Declared here with corank %d"_en_US, corank_);
        return false;
      }
    }
  } else { // Not a coarray
    if (hasAllocateCoarraySpec()) {
      // C941
      context.Say(name_.source,
          "Coarray specification must not appear in ALLOCATE when allocatable object is not a coarray"_err_en_US);
      return false;
    }
  }
  if (const parser::CoindexedNamedObject *
      coindexedObject{parser::GetCoindexedNamedObject(allocateObject_)}) {
    // C950
    context.Say(parser::FindSourceLocation(*coindexedObject),
        "Allocatable object must not be coindexed in ALLOCATE"_err_en_US);
    return false;
  }
  return true;
}

void AllocateChecker::Leave(const parser::AllocateStmt &allocateStmt) {
  if (auto info{CheckAllocateOptions(allocateStmt, context_)}) {
    for (const parser::Allocation &allocation :
        std::get<std::list<parser::Allocation>>(allocateStmt.t)) {
      AllocationCheckerHelper{allocation, *info}.RunChecks(context_);
    }
  }
}
} // namespace Fortran::semantics
