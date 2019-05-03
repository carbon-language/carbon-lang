// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "call.h"
#include "characteristics.h"
#include "expression.h"
#include "tools.h"
#include "../common/idioms.h"
#include "../semantics/symbol.h"

namespace Fortran::evaluate {

ActualArgument::ActualArgument(Expr<SomeType> &&x) : u_{std::move(x)} {}
ActualArgument::ActualArgument(common::CopyableIndirection<Expr<SomeType>> &&v)
  : u_{std::move(v)} {}
ActualArgument::ActualArgument(AssumedType x) : u_{x} {}
ActualArgument::~ActualArgument() {}

ActualArgument::AssumedType::AssumedType(const semantics::Symbol &symbol)
  : symbol_{&symbol} {
  const semantics::DeclTypeSpec *type{symbol.GetType()};
  CHECK(
      type != nullptr && type->category() == semantics::DeclTypeSpec::TypeStar);
}

int ActualArgument::AssumedType::Rank() const { return symbol_->Rank(); }

ActualArgument &ActualArgument::operator=(Expr<SomeType> &&expr) {
  u_ = std::move(expr);
  return *this;
}

std::optional<DynamicType> ActualArgument::GetType() const {
  if (const auto *expr{GetExpr()}) {
    return expr->GetType();
  } else {
    return std::nullopt;
  }
}

int ActualArgument::Rank() const {
  if (const auto *expr{GetExpr()}) {
    return expr->Rank();
  } else {
    return std::get<AssumedType>(u_).Rank();
  }
}

bool ActualArgument::operator==(const ActualArgument &that) const {
  return keyword == that.keyword &&
      isAlternateReturn == that.isAlternateReturn && u_ == that.u_;
}

SpecificIntrinsic::SpecificIntrinsic(
    IntrinsicProcedure n, characteristics::Procedure &&chars)
  : name{n}, characteristics{new characteristics::Procedure{std::move(chars)}} {
}

DEFINE_DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(SpecificIntrinsic)

SpecificIntrinsic::~SpecificIntrinsic() {}

bool SpecificIntrinsic::operator==(const SpecificIntrinsic &that) const {
  return name == that.name && characteristics == that.characteristics;
}

ProcedureDesignator::ProcedureDesignator(Component &&c)
  : u{common::CopyableIndirection<Component>::Make(std::move(c))} {}

std::optional<DynamicType> ProcedureDesignator::GetType() const {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    if (const auto &result{intrinsic->characteristics.value().functionResult}) {
      if (const auto *typeAndShape{result->GetTypeAndShape()}) {
        return typeAndShape->type();
      }
    }
  } else {
    return DynamicType::From(GetSymbol());
  }
  return std::nullopt;
}

int ProcedureDesignator::Rank() const {
  if (const Symbol * symbol{GetSymbol()}) {
    return symbol->Rank();
  }
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    if (const auto &result{intrinsic->characteristics.value().functionResult}) {
      if (const auto *typeAndShape{result->GetTypeAndShape()}) {
        CHECK(!typeAndShape->IsAssumedRank());
        return typeAndShape->Rank();
      }
    }
  }
  common::die("ProcedureDesignator::Rank(): no case");
  return 0;
}

bool ProcedureDesignator::IsElemental() const {
  if (const Symbol * symbol{GetSymbol()}) {
    return symbol->attrs().test(semantics::Attr::ELEMENTAL);
  }
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&u)}) {
    return intrinsic->characteristics.value().attrs.test(
        characteristics::Procedure::Attr::Elemental);
  }
  common::die("ProcedureDesignator::IsElemental(): no case");
  return 0;
}

const SpecificIntrinsic *ProcedureDesignator::GetSpecificIntrinsic() const {
  return std::get_if<SpecificIntrinsic>(&u);
}

const Component *ProcedureDesignator::GetComponent() const {
  if (auto *c{std::get_if<common::CopyableIndirection<Component>>(&u)}) {
    return &c->value();
  } else {
    return nullptr;
  }
}

const Symbol *ProcedureDesignator::GetSymbol() const {
  return std::visit(
      common::visitors{
          [](const Symbol *sym) { return sym; },
          [](const common::CopyableIndirection<Component> &c) {
            return &c.value().GetLastSymbol();
          },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      u);
}

parser::CharBlock ProcedureDesignator::GetName() const {
  return std::visit(
      common::visitors{
          [](const SpecificIntrinsic &i) -> parser::CharBlock {
            return i.name;
          },
          [](const Symbol *sym) -> parser::CharBlock { return sym->name(); },
          [](const common::CopyableIndirection<Component> &c)
              -> parser::CharBlock { return c.value().GetLastSymbol().name(); },
      },
      u);
}

Expr<SubscriptInteger> ProcedureRef::LEN() const {
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&proc_.u)}) {
    if (intrinsic->name == "repeat") {
      // LEN(REPEAT(ch,n)) == LEN(ch) * n
      CHECK(arguments_.size() == 2);
      const auto *stringArg{
          UnwrapExpr<Expr<SomeCharacter>>(arguments_[0].value())};
      const auto *nCopiesArg{
          UnwrapExpr<Expr<SomeInteger>>(arguments_[1].value())};
      CHECK(stringArg != nullptr && nCopiesArg != nullptr);
      auto stringLen{stringArg->LEN()};
      return std::move(stringLen) *
          ConvertTo(stringLen, common::Clone(*nCopiesArg));
    }
    if (intrinsic->name == "trim") {
      // LEN(TRIM(ch)) is unknown without execution.
      CHECK(arguments_.size() == 1);
      const auto *stringArg{
          UnwrapExpr<Expr<SomeCharacter>>(arguments_[0].value())};
      CHECK(stringArg != nullptr);
      return stringArg->LEN();
    }
  }
  return proc_.LEN();
}

FOR_EACH_SPECIFIC_TYPE(template class FunctionRef, )
}
