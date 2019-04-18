// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "check-allocate.h"
#include "attr.h"
#include "expression.h"
#include "tools.h"
#include "type.h"
#include "../evaluate/fold.h"
#include "../evaluate/type.h"
#include "../parser/parse-tree.h"

namespace Fortran::semantics {

struct AllocateCheckerInfo {
  const DeclTypeSpec *typeSpec{nullptr};
  std::optional<evaluate::DynamicType> sourceExprType;
  bool gotStat{false};
  bool gotMsg{false};
  bool gotTypeSpec{false};
  bool gotSrc{false};
  bool gotMold{false};
};

static const parser::Name &GetName(
    const parser::AllocateObject &allocateObject) {
  return std::visit(
      common::visitors{
          [&](const parser::Name &name) -> const parser::Name & {
            return name;
          },
          [&](const parser::StructureComponent &structureComponent)
              -> const parser::Name & { return structureComponent.component; },
      },
      allocateObject.u);
}

class AllocationChecker {
public:
  AllocationChecker(
      const parser::AllocateObject &obj, AllocateCheckerInfo &info)
    : allocateInfo_{info}, name_{GetName(obj)}, type_{name_.symbol->GetType()},
      isSubobject_{std::holds_alternative<parser::StructureComponent>(obj.u)} {
    CHECK(type_ != nullptr);
    if (type_->category() == DeclTypeSpec::Category::Character) {
      hasDefferedTypeParameter_ =
          type_->characterTypeSpec().length().isDeferred();
    } else if (const DerivedTypeSpec * derivedTypeSpec{type_->AsDerived()}) {
      for (auto it{derivedTypeSpec->parameters().begin()};
           it != derivedTypeSpec->parameters().end(); ++it) {
        hasDefferedTypeParameter_ |= it->second.isDeferred();
      }
      isAbstract_ = derivedTypeSpec->typeSymbol().attrs().test(Attr::ABSTRACT);
    }
    isUnlimitedPolymorphic_ =
        type_->category() == DeclTypeSpec::Category::ClassStar;
  }

  bool RunChecks(SemanticsContext &context);

private:
  AllocateCheckerInfo &allocateInfo_;
  const parser::Name &name_;
  const DeclTypeSpec *type_;
  bool isSubobject_;
  bool hasDefferedTypeParameter_{false};
  bool isUnlimitedPolymorphic_{false};
  bool isAbstract_{false};
};

static std::optional<AllocateCheckerInfo> CheckAllocateOptions(
    const parser::AllocateStmt &, SemanticsContext &);
static bool IsTypeCompatible(const DeclTypeSpec &, const DeclTypeSpec &);
static bool IsTypeCompatible(
    const DeclTypeSpec &type1, const evaluate::DynamicType &type2);
static bool HaveSameAssumedTypeParameters(
    const DeclTypeSpec &, const DeclTypeSpec &);
static bool HaveCompatibleKindParameters(
    const DeclTypeSpec &, const DeclTypeSpec &);

void AllocateChecker::Leave(const parser::AllocateStmt &allocateStmt) {
  if (auto info{CheckAllocateOptions(allocateStmt, context_)}) {
    for (const parser::Allocation &allocation :
        std::get<std::list<parser::Allocation>>(allocateStmt.t)) {
      AllocationChecker allocationChecker{
          std::get<parser::AllocateObject>(allocation.t), *info};
      allocationChecker.RunChecks(context_);
    }
  }
}

static std::optional<AllocateCheckerInfo> CheckAllocateOptions(
    const parser::AllocateStmt &allocateStmt, SemanticsContext &context) {
  AllocateCheckerInfo info;
  evaluate::ExpressionAnalyzer analyzer{context};  // can emit error messages
  bool stopCheckingAllocate{false};  // for errors that would lead to ambiguity
  info.gotTypeSpec =
      std::get<std::optional<parser::TypeSpec>>(allocateStmt.t).has_value();
  if (info.gotTypeSpec) {
    info.typeSpec = std::get<std::optional<parser::TypeSpec>>(allocateStmt.t)
                        .value()
                        .declTypeSpec;
  }

  const parser::Expr *parserSourceExpr{nullptr};
  for (const parser::AllocOpt &allocOpt :
      std::get<std::list<parser::AllocOpt>>(allocateStmt.t)) {
    std::visit(
        common::visitors{
            [&](const parser::StatOrErrmsg &statOrErr) {
              std::visit(
                  common::visitors{
                      [&](const parser::StatVariable &statVariable) {
                        analyzer.Analyze(statVariable.v);
                        if (info.gotStat) {  // C943
                          context.Say(
                              "STAT may not be duplicated in a ALLOCATE statement"_err_en_US);
                        }
                        info.gotStat = true;
                      },
                      [&](const parser::MsgVariable &msgVariable) {
                        analyzer.Analyze(msgVariable.v);
                        if (info.gotMsg) {  // C943
                          context.Say(
                              "ERRMSG may not be duplicated in a ALLOCATE statement"_err_en_US);
                        }
                        info.gotMsg = true;
                      },
                  },
                  statOrErr.u);
            },
            [&](const parser::AllocOpt::Source &source) {
              analyzer.Analyze(source.v);
              if (info.gotSrc) {  // C943
                context.Say(
                    "SOURCE may not be duplicated in a ALLOCATE statement"_err_en_US);
                stopCheckingAllocate = true;
              }
              if (info.gotMold || info.gotTypeSpec) {  // C944
                context.Say(
                    "At most one of source-expr and type-spec may appear in a ALLOCATE statement"_err_en_US);
                stopCheckingAllocate = true;
              }
              parserSourceExpr = &source.v.value();
              info.gotSrc = true;
            },
            [&](const parser::AllocOpt::Mold &mold) {
              analyzer.Analyze(mold.v);
              if (info.gotMold) {  // C943
                context.Say(
                    "MOLD may not be duplicated in a ALLOCATE statement"_err_en_US);
                stopCheckingAllocate = true;
              }
              if (info.gotSrc || info.gotTypeSpec) {  // C944
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

  if (info.gotSrc || info.gotMold) {
    CHECK(parserSourceExpr);
    CHECK(parserSourceExpr->typedExpr);  // TODO: Can we reach this spot without
                                         // a valid expression?
    info.sourceExprType = parserSourceExpr->typedExpr->v.GetType();
    if (!info.sourceExprType.has_value()) {
      context.Say(parserSourceExpr->source,
          "Source expression in ALLOCATE must be a valid expression"_err_en_US);
      return std::nullopt;
    }
  }
  return info;
}

bool AllocationChecker::RunChecks(SemanticsContext &context) {
  if (!IsVariableName(*name_.symbol)) {  // C932 pre-requisite
    context.Say(name_.source,
        "name in ALLOCATE statement must be a variable name"_err_en_US);
    return false;
  }
  if (!IsAllocatableOrPointer(*name_.symbol)) {  // C932
    context.Say(name_.source,
        "%s in ALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US,
        (isSubobject_ ? "component" : "name"));
    return false;
  }
  bool gotSourceExprOrTypeSpec{allocateInfo_.gotMold ||
      allocateInfo_.gotTypeSpec || allocateInfo_.gotSrc};
  if (hasDefferedTypeParameter_ && !gotSourceExprOrTypeSpec) {
    // C933
    context.Say(name_.source,
        "Either type-spec or source-expr shall appear in ALLOCATE when allocatable object has a deferred type parameters"_err_en_US);
    return false;
  }
  if (isUnlimitedPolymorphic_ && !gotSourceExprOrTypeSpec) {
    // C933
    context.Say(name_.source,
        "Either type-spec or source-expr shall appear in ALLOCATE when allocatable object is unlimited polymorphic"_err_en_US);
    return false;
  }
  if (isAbstract_ && !gotSourceExprOrTypeSpec) {
    // C933
    context.Say(name_.source,
        "Either type-spec or source-expr shall appear in ALLOCATE when allocatable object is of abstract type"_err_en_US);
    return false;
  }
  if (allocateInfo_.gotTypeSpec) {
    if (!IsTypeCompatible(*type_, *allocateInfo_.typeSpec)) {
      // C934
      context.Say(name_.source,
          "Allocatable object in ALLOCATE shall be type compatible with type-spec"_err_en_US);
      return false;
    }
    if (!HaveCompatibleKindParameters(*type_, *allocateInfo_.typeSpec)) {
      context.Say(name_.source,
          // C936
          "Kind type parameters of allocatable object in ALLOCATE shall be the same as the corresponding ones in type-spec"_err_en_US);
      return false;
    }
    if (!HaveSameAssumedTypeParameters(*type_, *allocateInfo_.typeSpec)) {
      // C935
      context.Say(name_.source,
          "Type parameters in type-spec shall be assumed if and only if they are assumed for allocatable object in ALLOCATE"_err_en_US);
      return false;
    }
  } else if (allocateInfo_.gotSrc || allocateInfo_.gotMold) {
    if (!IsTypeCompatible(*type_, allocateInfo_.sourceExprType.value())) {
      // first part of C945
      context.Say(name_.source,
          "Allocatable object in ALLOCATE shall be type compatible with source expression from MOLD or SOURCE"_err_en_US);
      return false;
    }
  }
  // TODO: Second part of C945, and C946. Shape related checks (C939, C940,
  // C942), Coarray related checks (C937, C941, C949, C950). Blacklisted type
  // checks (C938, C947, C948)
  return true;
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
      for (const DerivedTypeSpec *parent{&derivedType2}; parent != nullptr;
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
    // TypeStar does not make sens in allocate context because assumed type
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
    // TypeStar does not make sens in allocate context because assumed type
    // cannot be allocatable (C709)
    return true;
  }
  if (type2.category != evaluate::TypeCategory::Derived) {
    if (const IntrinsicTypeSpec * intrinsicType1{type1.AsIntrinsic()}) {
      return intrinsicType1->category() == type2.category;
    } else {
      return false;
    }
  } else {
    CHECK(type2.derived);
    return IsTypeCompatible(type1, *type2.derived);
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
    for (auto it{derivedType2->parameters().begin()};
         it != derivedType2->parameters().end(); ++it) {
      type2AssumedParametersCount += it->second.isAssumed();
    }
    if (const DerivedTypeSpec *
        derivedType1{
            type1.AsDerived()}) {  // type1 may be unlimited polymorphic
      for (auto it{derivedType1->parameters().begin()};
           it != derivedType1->parameters().end(); ++it) {
        if (it->second.isAssumed()) {
          ++type1AssumedParametersCount;
          if (const ParamValue *
              param{derivedType2->FindParameter(it->first)}) {
            if (!param->isAssumed()) {
              return false;  // type1 has an assumed param that is not assumed
                             // in type2
            }
          } else {
            return false;  // type1 has an assumed param that is not a type
                           // param of type2.
          }
        }
      }
    }
    // Will return false if type2 has type parameters that are not assumed in
    // type1 or do not exist in type1
    return type1AssumedParametersCount == type2AssumedParametersCount;
  }
  return true;  // other intrinsic types have no length type parameters
}

static std::optional<std::int64_t> GetTypeParameterInt64Value(
    const Symbol &parameterSymbol, const DerivedTypeSpec &derivedType) {
  if (const ParamValue *
      paramValue{derivedType.FindParameter(parameterSymbol.name())}) {
    return evaluate::ToInt64(paramValue->GetExplicit());
  } else {
    // Type parameter with default value and omitted in DerivedTypeSpec
    return evaluate::ToInt64(parameterSymbol.get<TypeParamDetails>().init());
  }
}

// Assumes type1 is type compatible with type2 (except for kind type parameters)
static bool HaveCompatibleKindParameters(
    const DeclTypeSpec &type1, const DeclTypeSpec &type2) {
  if (type1.category() == DeclTypeSpec::Category::ClassStar) {
    return true;
  }
  if (const IntrinsicTypeSpec * intrinsicType1{type1.AsIntrinsic()}) {
    const IntrinsicTypeSpec *intrinsicType2{type2.AsIntrinsic()};
    CHECK(intrinsicType2);  // Violation of type compatibility hypothesis.
    return intrinsicType1->kind() == intrinsicType2->kind();
  } else if (const DerivedTypeSpec * derivedType1{type1.AsDerived()}) {
    const DerivedTypeSpec *derivedType2{type2.AsDerived()};
    CHECK(derivedType2);  // Violation of type compatibility hypothesis.
    const DerivedTypeDetails &typeDetails{
        derivedType1->typeSymbol().get<DerivedTypeDetails>()};
    for (const Symbol *symbol :
        typeDetails.OrderParameterDeclarations(derivedType1->typeSymbol())) {
      if (symbol->get<TypeParamDetails>().attr() ==
          common::TypeParamAttr::Kind) {
        // At this point, it should have been ensured that these contain integer
        // constants, so die if this is not the case.
        if (GetTypeParameterInt64Value(*symbol, *derivedType1).value() !=
            GetTypeParameterInt64Value(*symbol, *derivedType2).value()) {
          return false;
        }
      }
    }
    return true;
  } else {
    common::die("unexpected type1 category");
  }
}

}
